import {
  Box,
  Heading,
  FormControl,
  FormLabel,
  FormHelperText,
  Button,
  VStack,
  useColorModeValue,
  Divider,
  HStack,
  RadioGroup,
  Radio,
  Stack,
  Select,
  Spinner,
  Text,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription
} from '@chakra-ui/react';
import { FaArrowLeft, FaCheck } from 'react-icons/fa';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSurvey } from '../../context/SurveyContext';

export default function ProjectStep({ onPrev }) {
  const navigate = useNavigate();
  const { surveyData, updateSurveyData, submitSurvey, loading, error } = useSurvey();
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  const [submitting, setSubmitting] = useState(false);
  
  // Handle code status radio change
  const handleCodeStatusChange = (value) => {
    // Reset all code status flags
    const codeStatus = {
      greenfield: 0,
      gpu_extend: 0,
      cpu_port: 0
    };
    
    // Set the selected one
    codeStatus[value] = 1;
    updateSurveyData(codeStatus);
  };
  
  // Get the current code status for radio group
  const getCodeStatus = () => {
    if (surveyData.greenfield === 1) return 'greenfield';
    if (surveyData.gpu_extend === 1) return 'gpu_extend';
    if (surveyData.cpu_port === 1) return 'cpu_port';
    return 'greenfield'; // Default
  };
  
  // Handle domain selection
  const handleDomainChange = (e) => {
    const domain = e.target.value;
    
    // Reset all domain flags
    const domainData = {
      domain_ai_ml: 0,
      domain_hpc: 0,
      domain_climate: 0,
      domain_embedded: 0,
      domain_graphics: 0,
      domain_data_analytics: 0,
      domain_other: 0
    };
    
    // Set the selected one
    if (domain in domainData) {
      domainData[domain] = 1;
    }
    
    updateSurveyData(domainData);
  };
  
  // Get current domain for select box
  const getCurrentDomain = () => {
    if (surveyData.domain_ai_ml === 1) return 'domain_ai_ml';
    if (surveyData.domain_hpc === 1) return 'domain_hpc';
    if (surveyData.domain_climate === 1) return 'domain_climate';
    if (surveyData.domain_embedded === 1) return 'domain_embedded';
    if (surveyData.domain_graphics === 1) return 'domain_graphics';
    if (surveyData.domain_data_analytics === 1) return 'domain_data_analytics';
    if (surveyData.domain_other === 1) return 'domain_other';
    return '';
  };
  
  // Handle form submission
  const handleSubmit = async () => {
    setSubmitting(true);
    const results = await submitSurvey();
    setSubmitting(false);
    
    if (results) {
      navigate('/results');
    }
  };

  return (
    <Box
      bg={cardBg}
      borderWidth="1px"
      borderColor={borderColor}
      borderRadius="lg"
      p={6}
      boxShadow="md"
      width="100%"
    >
      <VStack spacing={6} align="start">
        <Heading as="h2" size="lg" color="brand.500">
          Step 3: Project & Domain
        </Heading>
        
        <Divider />
        
        {/* Code Status */}
        <FormControl isRequired>
          <FormLabel fontSize="lg" fontWeight="bold">
            Code-base status
          </FormLabel>
          <FormHelperText mb={4}>
            What is the current state of your code?
          </FormHelperText>
          
          <RadioGroup 
            onChange={handleCodeStatusChange} 
            value={getCodeStatus()}
            colorScheme="brand"
          >
            <Stack direction="column" spacing={4}>
              <Radio value="greenfield">
                <Text fontWeight="medium">Brand-new project</Text>
                <Text fontSize="sm" color="gray.500">Starting from scratch</Text>
              </Radio>
              <Radio value="gpu_extend">
                <Text fontWeight="medium">Extending existing GPU code</Text>
                <Text fontSize="sm" color="gray.500">Adding features to code that already uses GPUs</Text>
              </Radio>
              <Radio value="cpu_port">
                <Text fontWeight="medium">Porting legacy CPU code</Text>
                <Text fontSize="sm" color="gray.500">Migrating CPU-only code to use accelerators</Text>
              </Radio>
            </Stack>
          </RadioGroup>
        </FormControl>
        
        {/* Application Domain */}
        <FormControl isRequired mt={6}>
          <FormLabel fontSize="lg" fontWeight="bold">
            Primary application domain
          </FormLabel>
          <FormHelperText mb={4}>
            What is the main purpose of your application?
          </FormHelperText>
          
          <Select 
            placeholder="Select domain" 
            value={getCurrentDomain()} 
            onChange={handleDomainChange}
            size="lg"
          >
            <option value="domain_ai_ml">AI/ML</option>
            <option value="domain_hpc">HPC simulation</option>
            <option value="domain_climate">Climate</option>
            <option value="domain_embedded">Embedded/Edge</option>
            <option value="domain_graphics">Graphics/Rendering</option>
            <option value="domain_data_analytics">Data analytics</option>
            <option value="domain_other">Other</option>
          </Select>
        </FormControl>
        
        {error && (
          <Alert status="error" mt={4} borderRadius="md">
            <AlertIcon />
            <Box>
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Box>
          </Alert>
        )}
        
        <HStack justifyContent="space-between" width="100%" mt={8} flexWrap={{ base: "wrap", md: "nowrap" }} gap={4}>
          <Button
            size={{ base: "md", lg: "lg" }}
            variant="outline"
            leftIcon={<FaArrowLeft />}
            onClick={onPrev}
            isDisabled={submitting}
            width={{ base: "100%", md: "auto" }}
          >
            Back
          </Button>
          <Button
            size={{ base: "md", lg: "lg" }}
            colorScheme="brand"
            rightIcon={submitting ? <Spinner size="sm" /> : <FaCheck />}
            onClick={handleSubmit}
            isLoading={submitting}
            loadingText="Analyzing..."
            width={{ base: "100%", md: "auto" }}
            fontSize={{ base: "sm", md: "md" }}
          >
            Get Results
          </Button>
        </HStack>
      </VStack>
    </Box>
  );
} 