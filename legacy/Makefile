# Compiler and flags
CC = gcc
CFLAGS = -O2 -Wall -lm -g -march=native -mavx

# Source files
SOURCES = argmax.c client4.c client5.c cumsum.c fft_avx.c fft_basic.c process.c saver.c server2.c server2b.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = radio_telescope

# Default build target
all: $(TARGET)

$(TARGET): $(OBJECTS)
    $(CC) $(CFLAGS) -o $@ $(OBJECTS)

%.o: %.c
    $(CC) $(CFLAGS) -c $< -o $@

clean:
    rm -f $(TARGET) $(OBJECTS)