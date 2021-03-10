IRPF90 = irpf90/bin/irpf90 --codelet=factor_een:2 #-s nelec:10 -s nnuc:2 -s ncord:5 #-a -d
FC     = ifort -xCORE-AVX512 -g -mkl=sequential
FCFLAGS= -O2 -I .
NINJA  = ninja
ARCHIVE = ar crs
RANLIB = ranlib

SRC=
OBJ=
LIB=

-include irpf90.make
export

irpf90.make: $(filter-out IRPF90_temp/%, $(wildcard */*.irp.f)) $(wildcard *.irp.f) $(wildcard *.inc.f) Makefile
	$(IRPF90)
