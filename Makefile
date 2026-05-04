CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O3 -march=native -D_POSIX_C_SOURCE=199309L -Iinclude
LDFLAGS = -lm

COMMON_SRC = src/vit.c src/vit_kernels.c src/vit_params.c src/utils.c
APP_SRC = src/main.c
TEST_SRC = test/test_main.c
HEADERS = include/vit.h include/vit_kernels.h include/vit_params.h include/utils.h

.PHONY: all app test clean

all: build/vit_app build/test_vit

app: build/vit_app
test: build/test_vit

build/vit_app: $(COMMON_SRC) $(APP_SRC) $(HEADERS)
	mkdir -p build
	$(CC) $(CFLAGS) $(COMMON_SRC) $(APP_SRC) -o build/vit_app $(LDFLAGS)

build/test_vit: $(COMMON_SRC) $(TEST_SRC) $(HEADERS)
	mkdir -p build
	$(CC) $(CFLAGS) $(COMMON_SRC) $(TEST_SRC) -o build/test_vit $(LDFLAGS)

clean:
	rm -rf build