#include "GPU.h"

using namespace std;

int main(int argc, char* argv[])
{
	if (!GPU::isAvailable())
	{
		printf("No GPU is available, exiting\n");
		return 0;
	}

	string platform = GPU::getPlatformName();
	printf("Deploying GPU platform: %s\n", platform.c_str());
	printf("Persistent block count: %d\n", GPU::getBlockCount());

	return 0;
}

