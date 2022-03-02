CC = g++
CFLAGS = -fPIC -shared -std=c++14 -Wall
INCLUDES = -I./include
#LIBS = -lvodh_sdk -L/usr/local/lib
LIBS = -lvodh_sdk -L.
SRCS = vODLA.cpp
MAIN = libvodla.so
DST = /usr/local/lib

all: $(MAIN)
	@echo Generated vODLA library	

debug: CFLAGS += -g -DDEBUG -DTIMING -DLOOP_CNT=1
debug: $(MAIN)

$(MAIN): $(SRCS)
		$(CC) $(SRCS) $(CFLAGS) $(INCLUDES) $(LIBS) -o $(MAIN)

clean:
		$(RM) $(MAIN)

install:
		install -m 755 $(MAIN) $(DST) 

# g++ vODLA.cpp -fPIC -shared -std=c++14 -I./include -lvodh_sdk -L. -o libvodla.so