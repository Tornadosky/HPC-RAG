// Framework logo paths and colors
export const frameworkInfo = {
  CUDA: {
    logo: 'https://upload.wikimedia.org/wikipedia/en/b/b9/Nvidia_CUDA_Logo.jpg',
    color: '#76B900',  // NVIDIA green
    description: 'NVIDIA\'s parallel computing platform and API model for NVIDIA GPUs'
  },
  HIP: {
    logo: 'https://raw.githubusercontent.com/ROCm/ROCm.github.io/develop/static/img/HIP.png',
    color: '#ED1C24',  // AMD red
    description: 'GPU programming platform by AMD that can run on both AMD and NVIDIA hardware'
  },
  OpenCL: {
    logo: 'https://www.khronos.org/assets/images/api_logos/opencl.svg',
    color: '#5586A4',  // OpenCL blue
    description: 'Open, royalty-free standard for cross-platform, parallel programming'
  },
  SYCL: {
    logo: 'https://www.khronos.org/assets/images/api_logos/sycl.svg',
    color: '#3F51B5',  // SYCL blue
    description: 'Single-source C++ programming model for heterogeneous systems'
  },
  RAJA: {
    logo: 'https://computing.llnl.gov/sites/default/files/styles/header_image/public/2021-07/RAJA-logo_horizontal_large_2.png',
    color: '#F39200',  // RAJA orange
    description: 'Portable performance portability layer for DOE applications'
  },
  Kokkos: {
    logo: 'https://raw.githubusercontent.com/kokkos/kokkos/master/doc/KokkosLogoExample.png',
    color: '#2C387E',  // Kokkos purple
    description: 'C++ performance portability ecosystem for HPC and scientific computing'
  },
  OpenACC: {
    logo: 'https://www.openacc.org/sites/default/files/inline-images/openacc-logo-hero.png',
    color: '#205081',  // OpenACC blue
    description: 'Directive-based parallel programming model for accelerators'
  },
  OpenMP: {
    logo: 'https://www.openmp.org/wp-content/uploads/openmp-header-logo-100h.png',
    color: '#2A7AB5',  // OpenMP blue
    description: 'API for multi-platform shared-memory parallel programming'
  }
};

// Fallback function to generate a color if no logo available
export function getFrameworkColor(framework) {
  if (frameworkInfo[framework]) {
    return frameworkInfo[framework].color;
  }
  
  // Generate a color based on the framework name (fallback)
  let hash = 0;
  for (let i = 0; i < framework.length; i++) {
    hash = framework.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  return `hsl(${hash % 360}, 70%, 50%)`;
} 