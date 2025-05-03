import { useEffect, useState } from 'react';
import { 
  Box, 
  Heading, 
  Text, 
  Image, 
  Flex, 
  Badge, 
  useColorModeValue,
  Spinner,
  Button 
} from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { frameworkInfo } from '../../utils/frameworkLogos';
import { DownloadIcon } from '@chakra-ui/icons';

const MotionBox = motion(Box);
const MotionFlex = motion(Flex);
const MotionHeading = motion(Heading);

export default function TopFrameworkCard({ topFramework, probability, explanation }) {
  const [showLogo, setShowLogo] = useState(false);
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const badgeColorScheme = useColorModeValue('brand', 'cyan');
  
  const framework = topFramework;
  const frameworkData = frameworkInfo[framework] || {
    logo: null,
    color: '#718096',
    description: 'A parallel programming framework'
  };
  
  // Animate the spinner -> logo transition
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowLogo(true);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <Box 
      bg={bgColor} 
      borderWidth="1px" 
      borderColor={borderColor} 
      borderRadius="lg"
      p={6}
      boxShadow="lg"
      width="100%"
      position="relative"
      overflow="hidden"
    >
      {/* Background accent */}
      <Box
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        width="30%"
        bgGradient={`linear(to-br, ${frameworkData.color}33, ${frameworkData.color}11)`}
        zIndex={0}
      />
      
      <MotionFlex 
        direction={{ base: 'column', md: 'row' }}
        align="center"
        justify="space-between"
        gap={6}
        zIndex={1}
        position="relative"
      >
        <Box>
          <Badge
            colorScheme={badgeColorScheme}
            mb={2}
            fontSize="sm"
            borderRadius="full"
            px={3}
            py={1}
          >
            Top Recommendation
          </Badge>
          
          <MotionHeading 
            as="h2" 
            size="xl" 
            mb={2}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            {framework}
          </MotionHeading>
          
          <MotionBox
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <Text fontSize="2xl" fontWeight="bold" color={frameworkData.color}>
              {probability > 1 ? Math.round(probability) : Math.round(probability * 100)}% match
            </Text>
            
            <Text fontSize="md" mt={4} maxW="600px">
              {explanation}
            </Text>
            
            <Text fontSize="sm" mt={4} color="gray.500">
              {frameworkData.description}
            </Text>
            
            <Button
              colorScheme="brand"
              size="md"
              mt={6}
              leftIcon={<DownloadIcon />}
              isDisabled
              _hover={{ cursor: 'not-allowed' }}
            >
              Download {framework} Starter Template
            </Button>
          </MotionBox>
        </Box>
        
        <Flex 
          justify="center" 
          align="center" 
          minW="150px" 
          minH="150px"
        >
          {!showLogo ? (
            <MotionBox
              initial={{ opacity: 1 }}
              animate={{ opacity: 0 }}
              transition={{ delay: 0.8 }}
            >
              <Spinner 
                size="xl" 
                thickness="4px"
                color="brand.500"
              />
            </MotionBox>
          ) : (
            <MotionBox
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2, type: 'spring' }}
            >
              {frameworkData.logo ? (
                <Image 
                  src={frameworkData.logo} 
                  alt={`${framework} logo`} 
                  maxW="150px"
                  maxH="100px"
                  objectFit="contain"
                  fallback={
                    <Box 
                      width="150px" 
                      height="100px" 
                      bg={frameworkData.color} 
                      color="white"
                      textAlign="center"
                      display="flex"
                      alignItems="center"
                      justifyContent="center"
                      fontWeight="bold"
                      fontSize="xl"
                      borderRadius="md"
                    >
                      {framework}
                    </Box>
                  }
                />
              ) : (
                <Box 
                  width="150px" 
                  height="100px" 
                  bg={frameworkData.color} 
                  color="white"
                  textAlign="center"
                  display="flex"
                  alignItems="center"
                  justifyContent="center"
                  fontWeight="bold"
                  fontSize="xl"
                  borderRadius="md"
                >
                  {framework}
                </Box>
              )}
            </MotionBox>
          )}
        </Flex>
      </MotionFlex>
    </Box>
  );
} 