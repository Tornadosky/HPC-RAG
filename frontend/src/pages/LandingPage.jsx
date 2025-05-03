import { 
  Box, 
  Button, 
  Container, 
  Heading, 
  Text, 
  VStack,
  useColorModeValue,
  Icon,
  Flex,
  SimpleGrid
} from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';
import { FaRocket, FaChartBar, FaLaptopCode, FaCogs } from 'react-icons/fa';

export default function LandingPage() {
  const bgGradient = useColorModeValue(
    'linear(to-b, brand.50, gray.50)',
    'linear(to-b, gray.900, gray.800)'
  );
  const cardBg = useColorModeValue('white', 'gray.800');
  const cardBorder = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box bgGradient={bgGradient} minH="calc(100vh - 140px)">
      <Container maxW="container.xl" pt={[10, 20]} pb={[10, 20]}>
        <VStack spacing={10} align="center" textAlign="center">
          <Heading 
            as="h1" 
            size="2xl" 
            fontWeight="bold"
            bgGradient="linear(to-r, brand.400, brand.600)" 
            bgClip="text"
          >
            Find Your Ideal HPC Framework
          </Heading>
          
          <Text fontSize="xl" maxW="800px">
            Answer a few quick questions about your hardware, priorities, and project, 
            and we'll instantly recommend the best parallel programming framework for your needs.
          </Text>
          
          <Button
            as={RouterLink}
            to="/survey"
            size="lg"
            colorScheme="brand"
            variant="solid"
            leftIcon={<FaRocket />}
            px={8}
            py={6}
            fontSize="lg"
            _hover={{
              transform: 'translateY(-2px)',
              boxShadow: 'lg',
            }}
            _active={{
              transform: 'translateY(0)',
              boxShadow: 'md',
            }}
          >
            Start the Survey
          </Button>
          
          <SimpleGrid columns={{ base: 1, md: 3 }} spacing={8} mt={12} w="full">
            <FeatureCard 
              icon={FaLaptopCode} 
              title="Expert Advice" 
              description="Based on real-world HPC engineering experience and performance data"
            />
            <FeatureCard 
              icon={FaChartBar} 
              title="Data-Driven" 
              description="Our ML model provides ranked recommendations with confidence scores"
            />
            <FeatureCard 
              icon={FaCogs} 
              title="Tailored Results" 
              description="Get a personalized recommendation based on your unique requirements"
            />
          </SimpleGrid>
        </VStack>
      </Container>
    </Box>
  );
}

function FeatureCard({ icon, title, description }) {
  const cardBg = useColorModeValue('white', 'gray.800');
  const cardBorder = useColorModeValue('gray.200', 'gray.700');
  
  return (
    <Flex
      direction="column"
      align="center"
      p={6}
      bg={cardBg}
      borderWidth="1px"
      borderColor={cardBorder}
      borderRadius="lg"
      boxShadow="md"
      textAlign="center"
      _hover={{
        transform: 'translateY(-5px)',
        boxShadow: 'lg',
        borderColor: 'brand.300',
      }}
      transition="all 0.3s"
    >
      <Icon as={icon} boxSize={12} color="brand.500" mb={4} />
      <Heading as="h3" size="md" mb={2}>
        {title}
      </Heading>
      <Text color={useColorModeValue('gray.600', 'gray.400')}>
        {description}
      </Text>
    </Flex>
  );
} 