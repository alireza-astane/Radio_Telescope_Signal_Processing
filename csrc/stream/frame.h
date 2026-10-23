/*
 * Wire format between receivers (rx_sim) and the correlator daemon (corrd).
 *
 * Each receiver opens one TCP connection and sends a stream of frames:
 *   rt_frame_header (32 bytes, little endian) followed by
 *   n_samples complex samples as interleaved signed 8-bit I/Q (2 * n_samples bytes).
 * Frames carry a sequence number so the correlator can align antennas.
 *
 * Input files for rx_sim (written by benchmarks/stream_tools.py):
 *   uint32 magic RT_FILE_MAGIC, uint32 n_ant, uint64 n_samples,
 *   then int8 IQ data laid out as [n_ant][n_samples][2].
 * Visibility dumps written by corrd:
 *   per dump: rt_dump_header then n_bl * nchan complex64 (float re, float im),
 *   baselines in numpy.triu_indices order.
 */
#ifndef RT_FRAME_H
#define RT_FRAME_H

#include <stdint.h>

#define RT_FRAME_MAGIC 0x31465452u /* "RTF1" */
#define RT_FILE_MAGIC 0x31445452u  /* "RTD1" */
#define RT_DUMP_MAGIC 0x31565452u  /* "RTV1" */

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint16_t antenna;
    uint16_t n_ant;
    uint64_t seq;
    uint64_t t_ns; /* sender timestamp, CLOCK_MONOTONIC */
    uint32_t n_samples;
    uint32_t reserved;
} rt_frame_header;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t n_bl;
    uint32_t nchan;
    uint32_t n_frames; /* frames integrated in this dump */
    uint64_t first_seq;
} rt_dump_header;

_Static_assert(sizeof(rt_frame_header) == 32, "frame header must be 32 bytes");
_Static_assert(sizeof(rt_dump_header) == 24, "dump header must be 24 bytes");

#endif
