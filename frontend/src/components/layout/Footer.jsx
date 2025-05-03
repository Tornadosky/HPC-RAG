import { Box, Text, Link, useColorModeValue } from '@chakra-ui/react';

export default function Footer() {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const textColor = useColorModeValue('gray.600', 'gray.400');

  return (
    <Box 
      as="footer" 
      bg={bgColor} 
      borderTop="1px" 
      borderColor={borderColor} 
      py={4} 
      px={4}
    >
      <Text 
        textAlign="center" 
        fontSize="sm" 
        color={textColor}
      >
        &copy; {new Date().getFullYear()} HPC Framework Recommender | Built with 
        <Link href="https://reactjs.org" color="brand.500" mx={1} isExternal>
          React
        </Link>
        and
        <Link href="https://fastapi.tiangolo.com" color="brand.500" mx={1} isExternal>
          FastAPI
        </Link>
      </Text>
    </Box>
  );
} 