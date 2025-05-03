import { Box, Progress, Flex, Text, useColorModeValue } from '@chakra-ui/react';
import { useSurvey } from '../../context/SurveyContext';

export default function ProgressBar() {
  const { step } = useSurvey();
  const progressPercentage = (step / 3) * 100;
  const accentColor = useColorModeValue('brand.500', 'brand.400');
  
  return (
    <Box mb={8} w="100%">
      <Flex justify="space-between" mb={2}>
        <Text fontWeight="medium" fontSize="sm">
          Progress
        </Text>
        <Text fontWeight="medium" fontSize="sm">
          {Math.round(progressPercentage)}%
        </Text>
      </Flex>
      
      <Progress 
        value={progressPercentage} 
        size="sm" 
        colorScheme="brand"
        borderRadius="full"
        hasStripe
        isAnimated
      />
      
      <Flex justify="space-between" width="100%" mt={1}>
        <Text fontSize="xs" color={step >= 1 ? accentColor : 'gray.500'}>
          Hardware
        </Text>
        <Text fontSize="xs" color={step >= 2 ? accentColor : 'gray.500'}>
          Priorities & Skills
        </Text>
        <Text fontSize="xs" color={step >= 3 ? accentColor : 'gray.500'}>
          Project & Domain
        </Text>
      </Flex>
    </Box>
  );
} 