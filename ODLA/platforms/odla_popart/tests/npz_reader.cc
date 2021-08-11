#include "cnpy.h"

int main()
{
    cnpy::npz_t one_npz = cnpy::npz_load("one.npz");
    std::cout << "Loaded the npz file" << std::endl;
    for(auto &npy_array : one_npz)
    {
        std::cout << npy_array.first << " <------> " << std::endl;
        std::size_t n = 1;
        for(auto &shape : npy_array.second.shape)
            n *= shape;
        for(int i = 0; i < n; i++)
        {
            if(i%32 == 0 && i)
                std::cout << std::endl;
            std::cout << *(npy_array.second.data<std::uint32_t>() + i) << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}