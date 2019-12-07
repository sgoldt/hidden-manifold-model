CC=$(CXX) -std=c++11 -Igtest/include -I.
CFLAGS = -Wall -pedantic -O3  -Xpreprocessor -fopenmp
MFLAGS = -larmadillo -lomp

.PHONY : clean distclean

%.o: %.cpp
	$(CC) -c $(CFLAGS) $<

%.exe: %.o
	$(CC) -o $@ $(CFLAGS) $< $(MFLAGS)


# ============================================================
# PHONY targets

clean :
	rm -f *.o core gmon.out *.gcno


