TARGET       = mpi-svm

CXX          = mpicxx

SRCDIR       = src
OBJDIR       = obj
BINDIR       = bin

TARGET       := $(BINDIR)/$(TARGET)

CXXFLAGS     = -Wall -O3 -I./$(SRCDIR) -std=c++20 -march=native -ffast-math

LDFLAGS      = -lm
WINLDFLAGS   = -lm

CXX_SOURCES  = $(wildcard $(SRCDIR)/*.cpp)

CXX_OBJECTS  = $(CXX_SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: all
all: remove $(TARGET) clean

$(TARGET): $(CXX_OBJECTS) $(BINDIR)
	$(CXX) $(CXX_OBJECTS) $(LDFLAGS) -o $@

windows: remove build_win clean

build_win: $(CXX_OBJECTS) $(BINDIR)
	$(CXX) $(CXX_OBJECTS) $(WINLDFLAGS) -o $(TARGET).exe

$(CXX_OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@


$(OBJDIR):
	mkdir -p $@

$(BINDIR):
	mkdir -p $@

clean:
	$(RM) -rf $(CXX_OBJECTS) $(OBJDIR)

remove:
	$(RM) -rf $(CXX_OBJECTS) $(OBJDIR) $(TARGET).exe $(TARGET) $(BINDIR)
