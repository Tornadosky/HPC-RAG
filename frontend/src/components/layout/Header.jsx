import { 
  Box, 
  Flex, 
  Heading, 
  useColorModeValue, 
  IconButton, 
  useColorMode 
} from '@chakra-ui/react';
import { MoonIcon, SunIcon } from '@chakra-ui/icons';
import { Link } from 'react-router-dom';

export default function Header() {
  const { colorMode, toggleColorMode } = useColorMode();
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box 
      as="header" 
      bg={bgColor} 
      borderBottom="1px" 
      borderColor={borderColor} 
      py={3} 
      px={4}
      position="sticky"
      top={0}
      zIndex={10}
      boxShadow="sm"
    >
      <Flex align="center" justify="space-between" maxW="container.xl" mx="auto">
        <Link to="/">
          <Heading 
            as="h1" 
            size="md" 
            color="brand.500"
            _hover={{ color: 'brand.600' }}
          >
            HPC Framework Recommender
          </Heading>
        </Link>
        
        <IconButton
          aria-label={`Switch to ${colorMode === 'light' ? 'dark' : 'light'} mode`}
          icon={colorMode === 'light' ? <MoonIcon /> : <SunIcon />}
          onClick={toggleColorMode}
          variant="ghost"
          color={useColorModeValue('gray.600', 'gray.400')}
          _hover={{
            bg: useColorModeValue('gray.100', 'gray.700')
          }}
        />
      </Flex>
    </Box>
  );
} 