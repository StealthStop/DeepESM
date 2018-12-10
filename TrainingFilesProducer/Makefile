# directory to put intermediate files 
ODIR                = obj
HDF5_DIRECTORY      = /cvmfs/cms.cern.ch/slc6_amd64_gcc630/external/hdf5/1.8.17-oenich/
HDF5_INC_DIRECTORY  = include

IDIR       = .
SDIR       = .
RSDIR      = $(CMSSW_BASE)/src/SusyAnaTools/Tools
STEALTHFWI = $(CMSSW_BASE)/src/Framework/Framework/include
STEALTHFWS = $(CMSSW_BASE)/src/Framework/Framework/src

CXX       = g++
LD        = g++
CXXFLAGS  = -g -O2 --std=c++11
LDFLAGS   = 
PREFIX    = /usr/local
LDCONFIG  = 
LIBSUFFIX = so
##End variables set by configure

INCLUDESDIRS  = -I. -I$(CMSSW_BASE)/src

# hdf5 include dir
INCLUDESDIRS  += -I$(HDF5_DIRECTORY)/$(HDF5_INC_DIRECTORY)

## SUSYAnaTools
INCLUDESDIRS += -I$(RSDIR) 

## Stealth Group
INCLUDESDIRS += -I$(STEALTHFWI) -I$(STEALTHFWS)

## Include ROOT
CXXFLAGS += $(shell root-config --cflags)

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

# Local Area
$(ODIR)/%.o : $(SDIR)/%.C
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(IDIR) $(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(SDIR)/%.cc
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(IDIR) $(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(SDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(IDIR) $(INCLUDESDIRS) -o $@ -c $<

# SusyAnaTools Repo
$(ODIR)/%.o : $(RSDIR)/%.C
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) $(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(RSDIR)/%.c
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) $(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(RSDIR)/%.cc
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) $(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(RSDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) $(INCLUDESDIRS) -o $@ -c $<

# Stealth Framework Repo
$(ODIR)/%.o : $(STEALTHFWS)/%.C
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) $(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(STEALTHFWS)/%.cc
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) $(INCLUDESDIRS) -o $@ -c $<

$(ODIR)/%.o : $(STEALTHFWS)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(ISDIR) $(INCLUDESDIRS) -o $@ -c $<

makeTrainingTuples: $(ODIR)/NTupleReader.o $(ODIR)/samples.o $(ODIR)/MiniTupleMaker.o $(ODIR)/makeTrainingTuples.o $(ODIR)/customize.o $(ODIR)/EventShapeVariables.o $(ODIR)/SATException.o
	$(LD) $^ $(LIBS) -o $@

clean:
	rm -rf $(ODIR)/*.o $(ODIR)/*.so $(ODIR)/*.d $(PROGRAMS) core $(ODIR)

-include $(ODIR)/*.d
