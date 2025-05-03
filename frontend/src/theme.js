import { extendTheme } from '@chakra-ui/react';

const config = {
  initialColorMode: 'light',
  useSystemColorMode: false,
};

const theme = extendTheme({
  config,
  fonts: {
    heading: 'Inter, sans-serif',
    body: 'Inter, sans-serif',
  },
  colors: {
    brand: {
      50: '#e0f7ff',
      100: '#b8e8ff',
      200: '#8cd8ff',
      300: '#5ec8ff',
      400: '#36b9ff',
      500: '#00aaff',  // Primary cyan accent
      600: '#0088cc',
      700: '#006699',
      800: '#004466',
      900: '#002233',
    }
  },
  styles: {
    global: (props) => ({
      body: {
        bg: props.colorMode === 'dark' ? 'gray.900' : 'gray.50',
        color: props.colorMode === 'dark' ? 'white' : 'gray.800',
      },
    }),
  },
  components: {
    Button: {
      baseStyle: {
        fontWeight: 'semibold',
        borderRadius: 'md',
      },
      variants: {
        primary: (props) => ({
          bg: 'brand.500',
          color: 'white',
          _hover: {
            bg: 'brand.600',
            _disabled: {
              bg: 'brand.500',
            },
          },
          _active: {
            bg: 'brand.700',
          },
        }),
      },
    },
    Progress: {
      baseStyle: {
        track: {
          borderRadius: 'md',
        },
        filledTrack: {
          borderRadius: 'md',
        },
      },
    },
  },
});

export default theme; 