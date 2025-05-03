import { useRef } from 'react';
import { Box, useColorModeValue, Text, VStack, HStack, Progress } from '@chakra-ui/react';
import { getFrameworkColor } from '../../utils/frameworkLogos';

export default function FrameworkBarChart({ ranking }) {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const textColor = useColorModeValue('gray.700', 'gray.200');

  console.log("FrameworkBarChart received ranking:", ranking);

  // Sort ranking by probability (descending)
  const sortedRanking = [...ranking].sort((a, b) => b.prob - a.prob);
  console.log("Sorted ranking:", sortedRanking);
  
  // Calculate percentages for display
  const frameworksWithPercentages = sortedRanking.map(item => {
    const percentage = item.prob > 1 ? Math.round(item.prob) : Math.round(item.prob * 100);
    console.log(`Framework: ${item.framework}, Raw prob: ${item.prob}, Calculated percentage: ${percentage}%`);
    return {
      ...item,
      percentage
    };
  });

  return (
    <Box 
      bg={bgColor} 
      borderWidth="1px" 
      borderColor={borderColor} 
      borderRadius="lg"
      p={6}
      boxShadow="md"
    >
      <VStack spacing={4} align="stretch">
        {frameworksWithPercentages.map(item => (
          <Box key={item.framework}>
            <HStack justify="space-between" mb={1}>
              <Text fontWeight="medium">{item.framework}</Text>
              <Text fontWeight="bold">{item.percentage}%</Text>
            </HStack>
            <Progress 
              value={item.percentage} 
              max={100}
              colorScheme="brand"
              height="24px"
              borderRadius="md"
              bgColor={useColorModeValue('gray.100', 'gray.700')}
              sx={{
                '& > div': {
                  backgroundColor: getFrameworkColor(item.framework),
                  transition: 'width 1s ease-in-out'
                }
              }}
            />
          </Box>
        ))}
      </VStack>
    </Box>
  );
} 