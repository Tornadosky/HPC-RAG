import { Box, Container, useColorModeValue } from '@chakra-ui/react';
import { useEffect } from 'react';
import { useSurvey } from '../context/SurveyContext';
import ProgressBar from '../components/survey/ProgressBar';
import HardwareStep from '../components/survey/HardwareStep';
import PrioritiesStep from '../components/survey/PrioritiesStep';
import ProjectStep from '../components/survey/ProjectStep';

export default function SurveyPage() {
  const { step, setStep } = useSurvey();
  const bgGradient = useColorModeValue(
    'linear(to-b, brand.50, gray.50)',
    'linear(to-b, gray.900, gray.800)'
  );

  useEffect(() => {
    // Scroll to top when step changes
    window.scrollTo(0, 0);
  }, [step]);

  const handleNext = () => {
    setStep(prev => Math.min(prev + 1, 3));
  };

  const handlePrev = () => {
    setStep(prev => Math.max(prev - 1, 1));
  };

  return (
    <Box py={10} bgGradient={bgGradient} minH="calc(100vh - 136px)">
      <Container maxW="container.md">
        <ProgressBar />
        
        {step === 1 && <HardwareStep onNext={handleNext} />}
        {step === 2 && <PrioritiesStep onNext={handleNext} onPrev={handlePrev} />}
        {step === 3 && <ProjectStep onPrev={handlePrev} />}
      </Container>
    </Box>
  );
} 