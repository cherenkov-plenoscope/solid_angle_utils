import setuptools


with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()


with open(os.path.join("solid_angle_utils", "version.py")) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")


setuptools.setup(
    name="solid_angle_utils_sebastian-achim-mueller",
    version=version,
    description="Helps you with solid angles and cones.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/solid_angle_utils",
    project_urls={
        "Bug Tracker": "https://github.com/cherenkov-plenoscope/solid_angle_utils/issues",
    },
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=["solid_angle_utils", "solid_angle_utils.cone",],
    package_data={"solid_angle_utils": []},
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
    ],
)
