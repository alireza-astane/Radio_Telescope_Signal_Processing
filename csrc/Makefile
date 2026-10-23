# Builds the shared library used by the Python package and the streaming tools.
CC      ?= gcc
CFLAGS  := -O3 -march=native -Wall -Wextra -std=gnu11 $(EXTRA_CFLAGS)
OMP     := -fopenmp
LIBDIR  := ../src/radiotel/correlator
LIB     := $(LIBDIR)/libfxcorr.so

all: $(LIB) stream

$(LIB): fxcorr.c fft.c fxcorr.h fft.h
	$(CC) $(CFLAGS) $(OMP) -fPIC -shared -o $@ fxcorr.c fft.c -lm

stream: stream/rx_sim stream/corrd

stream/rx_sim: stream/rx_sim.c stream/frame.h
	$(CC) $(CFLAGS) -o $@ stream/rx_sim.c -lpthread -lm

stream/corrd: stream/corrd.c stream/frame.h fxcorr.c fft.c
	$(CC) $(CFLAGS) $(OMP) -I. -o $@ stream/corrd.c fxcorr.c fft.c -lpthread -lm

clean:
	rm -f $(LIB) stream/rx_sim stream/corrd

.PHONY: all stream clean
