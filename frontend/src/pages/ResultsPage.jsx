import { 
  Box, 
  Container, 
  VStack, 
  Button, 
  Heading, 
  Text, 
  useColorModeValue,
  Alert,
  AlertIcon,
  AlertTitle,
  Spinner,
  Center,
  Divider
} from '@chakra-ui/react';
import { useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { RepeatIcon } from '@chakra-ui/icons';
import { useSurvey } from '../context/SurveyContext';
import TopFrameworkCard from '../components/results/TopFrameworkCard';
import FrameworkBarChart from '../components/results/FrameworkBarChart';
import ChatWidget from '../components/chat/ChatWidget';

export default function ResultsPage() {
  const navigate = useNavigate();
  const { results, resetSurvey, surveyData } = useSurvey();
  const bgGradient = useColorModeValue(
    'linear(to-b, brand.50, gray.50)',
    'linear(to-b, gray.900, gray.800)'
  );
  
  // Add logging for results data
  useEffect(() => {
    if (results) {
      console.log("ResultsPage received data:", results);
      console.log("Top framework:", results.ranking[0]);
      
      // Verify all probabilities
      results.ranking.forEach(item => {
        console.log(`ResultsPage - Framework: ${item.framework}, Probability: ${item.prob}, As percentage: ${item.prob > 1 ? item.prob : Math.round(item.prob * 100)}%`);
      });
    }
  }, [results]);
  
  // Redirect to survey page if no results
  useEffect(() => {
    if (!results) {
      navigate('/survey');
    }
  }, [results, navigate]);
  
  // Handle starting over
  const handleStartOver = () => {
    resetSurvey();
    navigate('/');
  };
  
  // If no results yet, show loading
  if (!results) {
    return (
      <Center py={20} minH="calc(100vh - 136px)">
        <VStack spacing={6}>
          <Spinner size="xl" color="brand.500" thickness="4px" />
          <Text fontSize="lg">Loading results...</Text>
        </VStack>
      </Center>
    );
  }

  // Get top framework
  const topFramework = results.ranking[0];
  console.log("Passing top framework to card:", topFramework);
  
  return (
    <Box py={10} bgGradient={bgGradient} minH="calc(100vh - 136px)">
      <Container maxW="container.xl">
        <VStack spacing={8} align="stretch">
          {/* Section: Top Recommendation */}
          <TopFrameworkCard 
            topFramework={topFramework.framework} 
            probability={topFramework.prob}
            explanation={results.explanation}
          />
          
          {/* Section: All Frameworks */}
          <Box mt={8}>
            <Heading as="h3" size="lg" mb={6}>
              All Frameworks Ranked
            </Heading>
            
            <FrameworkBarChart ranking={results.ranking} />
          </Box>
          
          {/* Chat Widget Section */}
          <Box mt={8}>
            <Heading as="h3" size="lg" mb={6}>
              Ask Questions About Frameworks
            </Heading>
            <Text mb={4}>
              Use our RAG-powered assistant to ask detailed questions about these frameworks and get answers based on research papers.
            </Text>
            <ChatWidget 
              userProfile={surveyData} 
              frameworkRanking={results.ranking} 
            />
          </Box>
          
          <Divider my={4} />
          
          {/* Start Over Button */}
          <Box textAlign="center" my={8}>
            <Button
              leftIcon={<RepeatIcon />}
              colorScheme="brand"
              size="lg"
              onClick={handleStartOver}
            >
              Start Over
            </Button>
          </Box>
        </VStack>
      </Container>
    </Box>
  );
} 