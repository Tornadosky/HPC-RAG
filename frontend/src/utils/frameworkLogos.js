// Framework logo paths and colors
export const frameworkInfo = {
  CUDA: {
    logo: '/images/cuda-logo.png',
    color: '#76B900',  // NVIDIA green
    description: 'NVIDIA\'s parallel computing platform and API model for NVIDIA GPUs'
  },
  HIP: {
    logo: '/images/hip-bg.png',  // Using the new HIP image
    color: '#ED1C24',  // AMD red
    description: 'GPU programming platform by AMD that can run on both AMD and NVIDIA hardware'
  },
  OpenCL: {
    logo: '/images/opencl-logo.png',
    color: '#5586A4',  // OpenCL blue
    description: 'Open, royalty-free standard for cross-platform, parallel programming'
  },
  SYCL: {
    logo: '/images/sycl-logo.png',
    color: '#3F51B5',  // SYCL blue
    description: 'Single-source C++ programming model for heterogeneous systems'
  },
  RAJA: {
    logo: '/images/raja-logo.webp',
    color: '#F39200',  // RAJA orange
    description: 'Portable performance portability layer for DOE applications'
  },
  Kokkos: {
    logo: '/images/kokkos-logo.png',
    color: '#2C387E',  // Kokkos purple
    description: 'C++ performance portability ecosystem for HPC and scientific computing'
  },
  OpenACC: {
    logo: '/images/openacc-logo.png',
    color: '#205081',  // OpenACC blue
    description: 'Directive-based parallel programming model for accelerators'
  },
  OpenMP: {
    logo: '/images/openmp-logo.png',
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