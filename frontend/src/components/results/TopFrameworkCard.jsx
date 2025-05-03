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
  
  // Generate background image path based on framework name
  const bgImagePath = `/images/${framework.toLowerCase()}-bg.png`;
  
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
      p={{ base: 4, md: 6 }}
      boxShadow="lg"
      width="100%"
      position="relative"
      overflow="hidden"
    >
      {/* Background image for all frameworks */}
      <Box
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        width={{ base: "30%", md: "25%" }}
        zIndex={0}
        backgroundImage={`url('${bgImagePath}')`}
        backgroundSize="cover"
        backgroundPosition="right center"
        backgroundRepeat="no-repeat"
        display={{ base: "none", sm: "block" }}
        // Fallback to gradient if image fails to load
        sx={{
          '&::after': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            backgroundImage: `linear-gradient(to bottom right, ${frameworkData.color}33, ${frameworkData.color}11)`,
            opacity: 0,
            transition: 'opacity 0.3s',
            zIndex: -1
          },
          '&.error::after': {
            opacity: 1
          }
        }}
        onError={(e) => e.target.classList.add('error')}
      />
      
      <MotionFlex 
        direction={{ base: 'column', md: 'row' }}
        align={{ base: "flex-start", md: "center" }}
        justify="space-between"
        gap={{ base: 4, md: 6 }}
        zIndex={1}
        position="relative"
      >
        <Box 
          width={{ base: "100%", md: "100%" }}
          mb={{ base: 0, sm: 0 }}
          pr={{ base: 0, sm: "30%", md: "30%" }}
        >
          <Badge
            colorScheme={badgeColorScheme}
            mb={2}
            fontSize={{ base: "xs", md: "sm" }}
            borderRadius="full"
            px={3}
            py={1}
          >
            Top Recommendation
          </Badge>
          
          <MotionHeading 
            as="h2" 
            size={{ base: "lg", md: "xl" }}
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
            <Text 
              fontSize={{ base: "xl", md: "2xl" }} 
              fontWeight="bold" 
              color={frameworkData.color}
            >
              {probability > 1 ? Math.round(probability) : Math.round(probability * 100)}% match
            </Text>
            
            <Text 
              fontSize={{ base: "sm", md: "md" }} 
              mt={4}
            >
              {explanation}
            </Text>
            
            <Text 
              fontSize={{ base: "xs", md: "sm" }} 
              mt={4} 
              color="gray.500"
            >
              {frameworkData.description}
            </Text>
            
            <Button
              colorScheme="brand"
              size={{ base: "sm", md: "md" }}
              mt={{ base: 4, md: 6 }}
              leftIcon={<DownloadIcon />}
              isDisabled
              _hover={{ cursor: 'not-allowed' }}
              display={{ base: "none", md: "flex" }}
            >
              Download {framework} Starter Template
            </Button>
          </MotionBox>
        </Box>
      </MotionFlex>
    </Box>
  );
} 