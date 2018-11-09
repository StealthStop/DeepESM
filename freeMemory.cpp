#include <vector>
#include <memory>
#include <unistd.h>

//-------------------------------------
// g++ -std=c++11 freeMemory.cpp -O3
// time ./a.out
//-------------------------------------

int main()
{
    using namespace std;
    vector<unique_ptr<int>> tvoptatm;
    //Number of gigs to allocate which then should be free(ish)
    const int NUMGIGS = 150;
    //Number of integers set for each loop 250,000,000 * 4(Bytes per integer) = 1 Gig used
    const int NUMINT = 250000000;

    for(int i = 0; i < NUMGIGS; ++i)
    {
        tvoptatm.emplace_back(new int[NUMINT]);
        for(int j = 0; j < NUMINT; ++j)
        {
            tvoptatm.back().get()[j] = 42;
        }
    }
}
