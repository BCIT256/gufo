#!/usr/bin/env python3
"""Build isolated diagnostic llama.cpp precision controls, never production goldens.

Requires Python 3.12+, host CMake/Ninja and a C++ compiler. The source archive
must match Gufo's pinned llama.cpp revision. No download or inference is run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

REVISION = "68d9053afd4f4d0752ced6187585f862355a40be"
ARCHIVE_SHA256 = "7ec6ef2e02c29a15879cd4c997b55976f1f0bbce4a042de15b71b376df52dc40"
F32_DOTS = r"""
#include "../ggml-quants.h"
// P4d diagnostic precision control only. Same packed weights and model graph;
// replace activation-quantized dots with block dequantization and F64 sums.
static unsigned int orn_f32_used = 0;
unsigned int orn_f32_control_used(void) { return __atomic_load_n(&orn_f32_used,__ATOMIC_RELAXED); }
#define ORN_F32_DOT(suffix, tag, block_size) \
static void orn_f32_##suffix(int n,float *s,size_t bs,const void *vx,size_t bx,const void *vy,size_t by,int nrc) { \
  (void)bs;(void)bx;(void)by; assert(nrc==1 && n%block_size==0); \
  unsigned int mask=1U<<tag; \
  if (!(__atomic_load_n(&orn_f32_used,__ATOMIC_RELAXED)&mask)) __atomic_fetch_or(&orn_f32_used,mask,__ATOMIC_RELAXED); \
  const block_##suffix *x=vx;const float *y=vy;float row[block_size];double sum=0.0; \
  for (int b=0;b<n/block_size;++b) {dequantize_row_##suffix(x+b,row,block_size); \
    for(int i=0;i<block_size;++i) sum+=(double)row[i]*(double)y[b*block_size+i];} \
  *s=(float)sum; \
}
ORN_F32_DOT(q8_0,GGML_TYPE_Q8_0,32)
ORN_F32_DOT(q5_K,GGML_TYPE_Q5_K,256)
ORN_F32_DOT(q6_K,GGML_TYPE_Q6_K,256)
ORN_F32_DOT(iq4_xs,GGML_TYPE_IQ4_XS,256)
#undef ORN_F32_DOT
"""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_cpu(text):
    marker = "static const struct ggml_type_traits_cpu type_traits_cpu"
    if text.count(marker) != 1:
        raise ValueError("unexpected pinned CPU trait table")
    text = text.replace(marker, F32_DOTS + "\n" + marker)
    for name, suffix in [("Q8_0", "q8_0"), ("Q5_K", "q5_K"),
                         ("Q6_K", "q6_K"), ("IQ4_XS", "iq4_xs")]:
        pattern = rf"(\[GGML_TYPE_{name}\] = \{{)(.*?)(\n    \}},)"
        match = re.search(pattern, text, re.S)
        if match is None:
            raise ValueError(f"missing {name} CPU trait")
        body, count = re.subn(r"(\.vec_dot\s*=\s*)\w+",
                             rf"\g<1>orn_f32_{suffix}", match[2])
        if count != 1:
            raise ValueError(f"unexpected {name} dot trait")
        body, count = re.subn(r"(\.vec_dot_type\s*=\s*)\w+",
                             r"\g<1>GGML_TYPE_F32", body)
        if count != 1:
            raise ValueError(f"unexpected {name} activation trait")
        text = text[:match.start(2)] + body + text[match.end(2):]
    return text


def control_dumper(text, unfused=False):
    if unfused:
        marker = "  cp.n_seq_max = 1;"
        if text.count(marker) != 1:
            raise ValueError("unexpected dumper context setup")
        text = text.replace(marker, marker + "\n"
                            "  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;\n"
                            "  cp.type_k = GGML_TYPE_F32;\n"
                            "  cp.type_v = GGML_TYPE_F32;")
    marker = "int main(int argc, char** argv) {"
    text = text.replace(marker,
                        'extern "C" unsigned int orn_f32_control_used(void);\n' + marker)
    marker = "  llama_batch_free(batch);"
    if text.count(marker) != 1:
        raise ValueError("unexpected dumper teardown")
    return text.replace(marker,
                        "  const unsigned int used=orn_f32_control_used();\n"
                        '  std::printf("F32 control formats used: 0x%x\\n",used);\n'
                        "  if (used != ((1U<<8)|(1U<<13)|(1U<<14)|(1U<<23))) return 3;\n"
                        + marker)


def reject_oracle_destination(output):
    """A new diagnostic directory must not replace the native oracle build."""
    resolved = output.resolve()
    root = Path(__file__).resolve().parents[3]
    forbidden = [root / "artifacts" / "reference" / "build-cpu",
                 root / "artifacts" / "reference" / "src"]
    for existing in forbidden:
        if resolved == existing or existing in resolved.parents:
            raise ValueError("refusing to write inside the existing native oracle tree")


def validate_sources(source_cpu, variants, quantized):
    text = source_cpu.read_text()
    needles = ["orn_f32_q8_0", "orn_f32_q5_K", "orn_f32_q6_K", "orn_f32_iq4_xs"]
    present = [name for name in needles if name in text]
    if quantized and present:
        raise ValueError("quantized control contains F32 dot replacements")
    if not quantized and present != needles:
        raise ValueError("precision control is missing declared F32 dots")
    if not quantized:
        for name, suffix in (("Q8_0", "q8_0"), ("Q5_K", "q5_K"),
                             ("Q6_K", "q6_K"), ("IQ4_XS", "iq4_xs")):
            pattern = rf"\[GGML_TYPE_{name}\] = \{{(.*?)\n    \}},"
            match = re.search(pattern, text, re.S)
            if match is None or "GGML_TYPE_F32" not in match[1]:
                raise ValueError(f"{name} activation type was not switched to F32")
            if f"orn_f32_{suffix}" not in match[1]:
                raise ValueError(f"{name} dot was not switched")
    for name, body in variants.items():
        if "llama_batch_free" not in body:
            raise ValueError(name + " lost its teardown")
        if quantized:
            if "orn_f32_control_used" in body:
                raise ValueError(name + " quantized dumper still checks the F32 counter")
        elif "((1U<<8)|(1U<<13)|(1U<<14)|(1U<<23))" not in body:
            raise ValueError(name + " does not require all four format bits")


def validate_manifest(path, archive_sha, quantized, built):
    data = json.loads(path.read_text())
    expected = {"scope", "reference_revision", "archive_sha256", "quantized_control",
                "cmake_flags", "cpu_source_sha256", "dumper_source_sha256", "built"}
    missing = expected - data.keys()
    if missing:
        raise ValueError("manifest missing " + ", ".join(sorted(missing)))
    if data["archive_sha256"] != archive_sha or data["reference_revision"] != REVISION:
        raise ValueError("manifest does not record the pinned archive")
    if data["quantized_control"] is not quantized or data["built"] is not built:
        raise ValueError("manifest flags do not match this invocation")
    if not isinstance(data["dumper_source_sha256"], dict) or not data["dumper_source_sha256"]:
        raise ValueError("manifest has no dumper hashes")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path, help="new isolated directory; never an existing oracle build")
    parser.add_argument("--quantized", action="store_true",
                        help="keep original dots, matching the diagnostic backend flags")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true",
                        help="write source/manifest only; no build or inference")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if sha256(args.archive) != ARCHIVE_SHA256:
        parser.error("source archive does not match the pinned SHA256")
    output = args.output.resolve()
    try:
        reject_oracle_destination(output)
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive) as archive:
        archive.extractall(output, filter="data")
    source = output / ("llama.cpp-" + REVISION)
    cpu = source / "ggml/src/ggml-cpu/ggml-cpu.c"
    if not args.quantized:
        cpu.write_text(patch_cpu(cpu.read_text()))
    dumper = Path(__file__).with_name("llama_logits.cpp").read_text()
    variants = {"llama_logits": dumper if args.quantized else control_dumper(dumper)}
    if not args.quantized:
        variants["llama_precise_logits"] = control_dumper(dumper, unfused=True)
    for name, text in variants.items():
        (output / (name + ".cpp")).write_text(text)
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DLLAMA_CURL=OFF",
             "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
             "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TOOLS=OFF",
             "-DBUILD_SHARED_LIBS=ON", "-DGGML_LLAMAFILE=OFF", "-DGGML_CPU_REPACK=OFF"]
    manifest = {
        "scope": "diagnostic only; no production oracle replacement or performance claim",
        "reference_revision": REVISION, "archive_sha256": ARCHIVE_SHA256,
        "quantized_control": args.quantized, "cmake_flags": flags,
        "cpu_source_sha256": sha256(cpu),
        "dumper_source_sha256": {name: sha256(output / (name + ".cpp")) for name in variants},
        "built": False,
    }
    manifest_path = output / "control-identity.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate_sources(cpu, variants, args.quantized)
    validate_manifest(manifest_path, ARCHIVE_SHA256, args.quantized, False)
    if args.prepare_only:
        print("Prepared and validated diagnostic sources; no build or inference was run.")
        return
    subprocess.run(["cmake", "-S", str(source), "-B", str(output / "build"),
                    "-G", "Ninja", *flags], check=True)
    subprocess.run(["cmake", "--build", str(output / "build"), "--target", "llama",
                    "--parallel", str(args.jobs)], check=True)
    libraries = output / "build/bin"
    for name in variants:
        subprocess.run(["g++", "-std=c++17", "-O2", "-I" + str(source / "include"),
                        "-I" + str(source / "ggml/include"), str(output / (name + ".cpp")),
                        "-L" + str(libraries), "-lllama", "-lggml", "-lggml-cpu", "-lggml-base",
                        "-Wl,-rpath," + str(libraries), "-o", str(output / name)], check=True)
    manifest.update(built=True,
                    binary_sha256={name: sha256(output / name) for name in variants},
                    library_sha256={p.name: sha256(p) for p in sorted(libraries.glob("*.so"))},
                    compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                    cmake=subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    written = validate_manifest(manifest_path, ARCHIVE_SHA256, args.quantized, True)
    for name, digest in written["binary_sha256"].items():
        if digest != sha256(output / name):
            raise ValueError("manifest binary hash does not match " + name)
    print("Built and validated isolated diagnostic tools; no inference or qualification was run.")


if __name__ == "__main__":
    main()
