from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="loopt4_ext",
    ext_modules=[
        CUDAExtension(
            name="loopt4_ext",
            sources=[
                "csrc/bindings.cpp",
                "csrc/nvfp4_mxfp8_moe_sm75.cu",
                "csrc/attnres_sconv_sm75.cu",
                "csrc/hybrid_attn_sm75.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode=arch=compute_75,code=sm_75",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
