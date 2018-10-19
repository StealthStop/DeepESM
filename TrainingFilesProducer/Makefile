# directory to put intermediate files 
ODIR                = obj
HDF5_DIRECTORY      = /cvmfs/cms.cern.ch/slc6_amd64_gcc630/external/hdf5/1.8.17-oenich/
HDF5_INC_DIRECTORY  = include
SUSYANATOOLSDIR     = $(CMSSW_BASE)/src/SusyAnaTools
STEALTHFWDIR        = $(CMSSW_BASE)/src/Framework

IDIR       = .
SDIR       = .
RSDIR      = $(SUSYANATOOLSDIR)/Tools
RIDIR      = $(SUSYANATOOLSDIR)/Tools
STEALTHFWI = $(STEALTHFWDIR)/Framework/include
STEALTHFWS = $(STEALTHFWDIR)/Framework/src

CXX       = g++
LD        = g++
CXXFLAGS  = -g -O2 --std=c++11
LDFLAGS   = 
PREFIX    = /usr/local
LDCONFIG  = 
LIBSUFFIX = so
##End variables set by configure

# Top tagger includes
INCLUDESDIRS  += -I. 

#hdf5 include dir
INCLUDESDIRS  += -I$(HDF5_DIRECTORY)/$(HDF5_INC_DIRECTORY)

## Stealth Group
INCLUDESDIRS += -I$(STEALTHFWI) -I$(STEALTHFWS)

## Include ROOT
INCLUDESDIRS  += $(shell root-config --cflags)

## Enable for maximum warning
#CXXFLAGS += -Wall -Wextra -Weffc++

# Flags for generating auto dependancies 
CXXDEPFLAGS = -MMD -MP

#Necessary libraries, the order matters here!
# ROOT libraries
LIBS       = $(shell root-config --glibs)

# hdf5 libs
LIBS += -L$(HDF5_DIRECTORY)/lib -lhdf5

PROGRAMS = makeTrainingTuples 

all: mkobj sampPyWrap $(PROGRAMS)

mkobj:
	@mkdir -p obj

#code to compile shared library to link samples to python                                                                                                                               
sampPyWrap: $(ODIR)/samplesModule.so

$(ODIR)/samplesModule.so: $(ODIR)/samplesPyWrap.o $(ODIR)/samplesModulePyWrap.o
	$(CXX) -shared -o $@ $^

$(ODIR)/samplesPyWrap.o: $(RSDIR)/samples.cc $(RSDIR)/samples.h
	$(CXX) --std=c++11 -c -fPIC -o $@ $<

$(ODIR)/samplesModulePyWrap.o: $(RSDIR)/samplesModule.cc
	$(CXX) --std=c++11 -c -fPIC -o $@ $<


$(ODIR)/%.o : $(SDIR)/%.C
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(IDIR) -I$(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(SDIR)/%.cc
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(IDIR) -I$(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(SDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(IDIR) -I$(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(RSDIR)/%.C
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) -I$(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(RSDIR)/%.cc
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) -I$(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(RSDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) -I$(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(STEALTHFWS)/%.C
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) -I$(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(STEALTHFWS)/%.cc
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) -I$(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(STEALTHFWS)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) -I$(INCLUDESDIRS) -o $@ -c $<

makeTrainingTuples: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o $(ODIR)/MiniTupleMaker.o $(ODIR)/makeTrainingTuples.o $(ODIR)/customize.o $(ODIR)/EventShapeVariables.o
	$(LD) $^ $(LIBS) -o $@

clean:
	rm -f $(ODIR)/*.o $(ODIR)/*.so $(ODIR)/*.d $(PROGRAMS) core 
