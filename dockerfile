FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    gdb \
    git \
    libgtest-dev \
    gcovr \
    libopenblas-dev \
    locales \
    && rm -rf /var/lib/apt/lists/*

# generate the locale lcov's Perl runtime expects
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN cmake -S /usr/src/googletest -B /tmp/gtest-build \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/gtest-build \
    && cmake --install /tmp/gtest-build \
    && rm -rf /tmp/gtest-build

RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /workspace