# Build
docker image build . -t k_means_constrained

# Run interactive shell
docker run -v $(pwd):/opt/project -it k_means_constrained:latest bash

