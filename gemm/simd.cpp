#include <stdio.h>
#include <stdint.h>
#include "timer.h"

#include "cblas.h"

#if !defined(DGEMM_PAGE_SIZE)
#define DGEMM_PAGE_SIZE 4096
#endif

#if !defined(MMATX_SIZER)
#define MMATX_SIZER 528
#endif

#if !defined(MMATX_SIZEC)
#define MMATX_SIZEC 864
#endif

#if !defined(REP_EXP)
#define REP_EXP 1
#endif

#if !defined(DMATX_SIZER)
#define DMATX_SIZER 2112
#endif

#if !defined(DMATX_SIZEC)
#define DMATX_SIZEC 864
#endif

#if !defined(REP_EXP)
#define REP_EXP 1
#endif

double ma[MMATX_SIZER][MMATX_SIZEC] __attribute__ ((aligned(DGEMM_PAGE_SIZE)));
double mb[MMATX_SIZER][MMATX_SIZEC] __attribute__ ((aligned(DGEMM_PAGE_SIZE)));
double mc[MMATX_SIZER][MMATX_SIZEC] __attribute__ ((aligned(DGEMM_PAGE_SIZE)));

double da[DMATX_SIZER][DMATX_SIZEC] __attribute__ ((aligned(DGEMM_PAGE_SIZE)));
double db[DMATX_SIZER][DMATX_SIZEC] __attribute__ ((aligned(DGEMM_PAGE_SIZE)));
double dc[DMATX_SIZER][DMATX_SIZEC] __attribute__ ((aligned(DGEMM_PAGE_SIZE)));

#include "mipp.h"

////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm kernel window of 2x16

static void matmulm(
	double (&ma)[MMATX_SIZER][MMATX_SIZEC],
	double (&mb)[MMATX_SIZER][MMATX_SIZEC],
	double (&mc)[MMATX_SIZER][MMATX_SIZEC]) {

  double *ma2 = &(ma[0][0]);
  double *mb2 = &(mb[0][0]);
  double *mc2 = &(mc[0][0]);

  mipp::Reg<double> mmc_0_0;
  mipp::Reg<double> mmc_0_4;
  mipp::Reg<double> mmc_0_8;
  mipp::Reg<double> mmc_0_12;

  mipp::Reg<double> mmc_1_0;
  mipp::Reg<double> mmc_1_4;
  mipp::Reg<double> mmc_1_8;
  mipp::Reg<double> mmc_1_12;

  mipp::Reg<double> mmb0;
  mipp::Reg<double> mmb4;
  mipp::Reg<double> mmb8;
  mipp::Reg<double> mmb12;

  mipp::Reg<double> mma0_ji;
  mipp::Reg<double> mma1_ji;

	for (size_t j = 0; j < MMATX_SIZER; j += 2) {
		for (size_t k = 0; k < MMATX_SIZEC; k += mipp::N<double>()*8) {

			mmc_0_0.load(mc2 + (j + 0) * MMATX_SIZEC + (k + 0));
			mmc_0_4.load(mc2 + (j + 0) * MMATX_SIZEC + (k + mipp::N<double>()));
			mmc_0_8.load(mc2 + (j + 0) * MMATX_SIZEC + (k + mipp::N<double>()*2));
			mmc_0_12.load(mc2 + (j + 0) * MMATX_SIZEC + (k + mipp::N<double>()*3));

			mmc_1_0.load(mc2 + (j + 1) * MMATX_SIZEC + (k + 0));
			mmc_1_4.load(mc2 + (j + 1) * MMATX_SIZEC + (k + mipp::N<double>()));
			mmc_1_8.load(mc2 + (j + 1) * MMATX_SIZEC + (k + mipp::N<double>()*2));
			mmc_1_12.load(mc2 + (j + 1) * MMATX_SIZEC + (k + mipp::N<double>()*3));

			for (size_t i = 0; i < MMATX_SIZEC; ++i) {

				mmb0.load(mb2 + (i + 0) * MMATX_SIZEC + (k + 0));
				mmb4.load(mb2 + (i + 0) * MMATX_SIZEC + (k + mipp::N<double>()));
				mmb8.load(mb2 + (i + 0) * MMATX_SIZEC + (k + mipp::N<double>()*2));
				mmb12.load(mb2 + (i + 0) * MMATX_SIZEC + (k + mipp::N<double>()*3));

				mma0_ji.load(ma2 + (j + 0) * MMATX_SIZEC + i);
				mma0_ji.load(ma2 + (j + 1) * MMATX_SIZEC + i);

				mmc_0_0  += mma0_ji * mmb0;
				mmc_0_4  += mma0_ji * mmb4;
				mmc_0_8  += mma0_ji * mmb8;
				mmc_0_12 += mma0_ji * mmb12;

				mmc_1_0  += mma1_ji * mmb0;
				mmc_1_4  += mma1_ji * mmb4;
				mmc_1_8  += mma1_ji * mmb8;
				mmc_1_12 += mma1_ji * mmb12;
			}

			mmc_0_0.store(mc2 + (j + 0) * MMATX_SIZEC + (k + 0));
			mmc_0_4.store(mc2 + (j + 0) * MMATX_SIZEC + (k + mipp::N<double>()));
			mmc_0_8.store(mc2 + (j + 0) * MMATX_SIZEC + (k + mipp::N<double>()*2));
			mmc_0_12.store(mc2 + (j + 0) * MMATX_SIZEC + (k + mipp::N<double>()*3));

			mmc_1_0.store(mc2 + (j + 1) * MMATX_SIZEC + (k + 0));
			mmc_1_4.store(mc2 + (j + 1) * MMATX_SIZEC + (k + mipp::N<double>()));
			mmc_1_8.store(mc2 + (j + 1) * MMATX_SIZEC + (k + mipp::N<double>()*2));
			mmc_1_12.store(mc2 + (j + 1) * MMATX_SIZEC + (k + mipp::N<double>()*3));

		}
	}
}

static void matmuld(
	double (&da)[DMATX_SIZER][DMATX_SIZEC],
	double (&db)[DMATX_SIZER][DMATX_SIZEC],
	double (&dc)[DMATX_SIZER][DMATX_SIZEC]) {

  double *da2 = &(da[0][0]);
  double *db2 = &(db[0][0]);
  double *dc2 = &(dc[0][0]);

  mipp::Reg<double> mdc_0_0;
  mipp::Reg<double> mdc_0_4;
  mipp::Reg<double> mdc_0_8;
  mipp::Reg<double> mdc_0_12;

  mipp::Reg<double> mdc_1_0;
  mipp::Reg<double> mdc_1_4;
  mipp::Reg<double> mdc_1_8;
  mipp::Reg<double> mdc_1_12;

  mipp::Reg<double> mdb0;
  mipp::Reg<double> mdb4;
  mipp::Reg<double> mdb8;
  mipp::Reg<double> mdb12;

  mipp::Reg<double> mda0_ji;
  mipp::Reg<double> mda1_ji;

	for (size_t j = 0; j < DMATX_SIZER; j += 2) {
		for (size_t k = 0; k < DMATX_SIZEC; k += mipp::N<double>()*8) {

			mdc_0_0.load(dc2 + (j + 0) * DMATX_SIZEC + (k + 0));
			mdc_0_4.load(dc2 + (j + 0) * DMATX_SIZEC + (k + mipp::N<double>()));
			mdc_0_8.load(dc2 + (j + 0) * DMATX_SIZEC + (k + mipp::N<double>()*2));
			mdc_0_12.load(dc2 + (j + 0) * DMATX_SIZEC + (k + mipp::N<double>()*3));

			mdc_1_0.load(dc2 + (j + 1) * DMATX_SIZEC + (k + 0));
			mdc_1_4.load(dc2 + (j + 1) * DMATX_SIZEC + (k + mipp::N<double>()));
			mdc_1_8.load(dc2 + (j + 1) * DMATX_SIZEC + (k + mipp::N<double>()*2));
			mdc_1_12.load(dc2 + (j + 1) * DMATX_SIZEC + (k + mipp::N<double>()*3));

			for (size_t i = 0; i < DMATX_SIZEC; ++i) {

				mdb0.load(db2 + (i + 0) * DMATX_SIZEC + (k + 0));
				mdb4.load(db2 + (i + 0) * DMATX_SIZEC + (k + mipp::N<double>()));
				mdb8.load(db2 + (i + 0) * DMATX_SIZEC + (k + mipp::N<double>()*2));
				mdb12.load(db2 + (i + 0) * DMATX_SIZEC + (k + mipp::N<double>()*3));

				mda0_ji.load(da2 + (j + 0) * DMATX_SIZEC + i);
				mda0_ji.load(da2 + (j + 1) * DMATX_SIZEC + i);

				mdc_0_0  += mda0_ji * mdb0;
				mdc_0_4  += mda0_ji * mdb4;
				mdc_0_8  += mda0_ji * mdb8;
				mdc_0_12 += mda0_ji * mdb12;

				mdc_1_0  += mda1_ji * mdb0;
				mdc_1_4  += mda1_ji * mdb4;
				mdc_1_8  += mda1_ji * mdb8;
				mdc_1_12 += mda1_ji * mdb12;
			}

			mdc_0_0.store(dc2 + (j + 0) * DMATX_SIZEC + (k + 0));
			mdc_0_4.store(dc2 + (j + 0) * DMATX_SIZEC + (k + mipp::N<double>()));
			mdc_0_8.store(dc2 + (j + 0) * DMATX_SIZEC + (k + mipp::N<double>()*2));
			mdc_0_12.store(dc2 + (j + 0) * DMATX_SIZEC + (k + mipp::N<double>()*3));

			mdc_1_0.store(dc2 + (j + 1) * DMATX_SIZEC + (k + 0));
			mdc_1_4.store(dc2 + (j + 1) * DMATX_SIZEC + (k + mipp::N<double>()));
			mdc_1_8.store(dc2 + (j + 1) * DMATX_SIZEC + (k + mipp::N<double>()*2));
			mdc_1_12.store(dc2 + (j + 1) * DMATX_SIZEC + (k + mipp::N<double>()*3));

		}
	}
}

void run_simd_dgemmc(double *A, double *B, double *C) {
	for (size_t i = 0; i < MMATX_SIZER; ++i) {
		for (size_t j = 0; j < MMATX_SIZEC; ++j) {
		  ma[i][j] = A[j * MMATX_SIZER + i];
		  mb[i][j] = B[j * MMATX_SIZER + i];
    }
	}

	const uint64_t t0 = timer_ns();

  matmulm(ma, mb, mc);

	const uint64_t dt = timer_ns() - t0;

//#if PRINT_MATX != 0
//	fprint_matx(stdout, mc);
//
//#endif
//	fprintf(stdout, "%f\n", 1e-9 * dt);
	for (size_t i = 0; i < MMATX_SIZER; ++i) {
		for (size_t j = 0; j < MMATX_SIZEC; ++j) {
		  C[j * MMATX_SIZER + i] = mc[i][j];
    }
	}
}

void run_simd_dgemmd(double *A, double *B, double *C) {
	for (size_t i = 0; i < DMATX_SIZER; ++i) {
		for (size_t j = 0; j < DMATX_SIZEC; ++j) {
		  da[i][j] = A[j * DMATX_SIZER + i];
		  db[i][j] = B[j * DMATX_SIZER + i];
    }
	}

	const uint64_t t0 = timer_ns();

  matmuld(da, db, dc);

	const uint64_t dt = timer_ns() - t0;

//#if PRINT_MATX != 0
//	fprint_datx(stdout, dc);
//
//#endif
//	fprintf(stdout, "%f\n", 1e-9 * dt);
	for (size_t i = 0; i < DMATX_SIZER; ++i) {
		for (size_t j = 0; j < DMATX_SIZEC; ++j) {
		  C[j * DMATX_SIZER + i] = dc[i][j];
    }
	}
}

extern "C" {
  void run_simd_dgemmc_c(double *A, double *B, double *C, int m, int n, int k){run_simd_dgemmc(A, B, C);};
  void run_simd_dgemmd_c(double *A, double *B, double *C, int m, int n, int k){run_simd_dgemmd(A, B, C);};
}
