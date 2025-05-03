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
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  SliderMark,
  Tooltip,
  ButtonGroup,
  Icon,
  Wrap,
  WrapItem,
  Tag,
  TagLabel,
  TagLeftIcon,
  Text
} from '@chakra-ui/react';
import { useState } from 'react';
import { FaArrowLeft, FaArrowRight, FaBolt, FaExchangeAlt, FaPuzzlePiece, FaCode } from 'react-icons/fa';
import { GrDirections } from 'react-icons/gr';
import { VscSymbolNamespace } from 'react-icons/vsc';
import { useSurvey } from '../../context/SurveyContext';

export default function PrioritiesStep({ onNext, onPrev }) {
  const { surveyData, updateSurveyData } = useSurvey();
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // States for slider tooltips
  const [perfShowTooltip, setPerfShowTooltip] = useState(false);
  const [portShowTooltip, setPortShowTooltip] = useState(false);
  const [ecoShowTooltip, setEcoShowTooltip] = useState(false);
  const [lockinShowTooltip, setLockinShowTooltip] = useState(false);
  
  // Handle sliders
  const handlePerfWeightChange = (value) => {
    updateSurveyData({ perf_weight: value / 5 });
  };

  const handlePortWeightChange = (value) => {
    updateSurveyData({ port_weight: value / 5 });
  };

  const handleEcoWeightChange = (value) => {
    updateSurveyData({ eco_weight: value / 5 });
  };
  
  const handleLockinToleranceChange = (value) => {
    updateSurveyData({ lockin_tolerance: value / 100 });
  };
  
  // Handle skill level
  const handleSkillLevelChange = (level) => {
    updateSurveyData({ gpu_skill_level: level });
  };

  // Handle programming model preference
  const toggleDirectives = () => {
    updateSurveyData({ pref_directives: surveyData.pref_directives === 1 ? 0 : 1 });
  };
  
  const toggleKernels = () => {
    updateSurveyData({ pref_kernels: surveyData.pref_kernels === 1 ? 0 : 1 });
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
          Step 2: Priorities & Skills
        </Heading>
        
        <Divider />
        
        {/* Performance Weight */}
        <FormControl>
          <FormLabel fontSize="lg" fontWeight="bold">
            <HStack>
              <Icon as={FaBolt} color="yellow.500" />
              <Text>Raw performance on the primary device</Text>
            </HStack>
          </FormLabel>
          <FormHelperText mb={2}>
            How important is achieving maximum performance on your primary hardware?
          </FormHelperText>
          
          <Slider
            id="perf-slider"
            defaultValue={surveyData.perf_weight * 5}
            min={1}
            max={5}
            step={1}
            onChange={handlePerfWeightChange}
            onMouseEnter={() => setPerfShowTooltip(true)}
            onMouseLeave={() => setPerfShowTooltip(false)}
            mt={6}
            mb={10}
          >
            <SliderMark value={1} mt={2} ml={-2.5} fontSize="sm">
              Low
            </SliderMark>
            <SliderMark value={5} mt={2} ml={-10} fontSize="sm">
              Critical
            </SliderMark>
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <Tooltip
              hasArrow
              bg="brand.500"
              color="white"
              placement="top"
              isOpen={perfShowTooltip}
              label={`${Math.round(surveyData.perf_weight * 100)}%`}
            >
              <SliderThumb boxSize={6} />
            </Tooltip>
          </Slider>
        </FormControl>
        
        {/* Portability Weight */}
        <FormControl>
          <FormLabel fontSize="lg" fontWeight="bold">
            <HStack>
              <Icon as={FaExchangeAlt} color="green.500" />
              <Text>Portability across devices/vendors</Text>
            </HStack>
          </FormLabel>
          <FormHelperText mb={2}>
            How important is being able to run your code on different hardware vendors?
          </FormHelperText>
          
          <Slider
            id="port-slider"
            defaultValue={surveyData.port_weight * 5}
            min={1}
            max={5}
            step={1}
            onChange={handlePortWeightChange}
            onMouseEnter={() => setPortShowTooltip(true)}
            onMouseLeave={() => setPortShowTooltip(false)}
            mt={6}
            mb={10}
          >
            <SliderMark value={1} mt={2} ml={-2.5} fontSize="sm">
              Low
            </SliderMark>
            <SliderMark value={5} mt={2} ml={-10} fontSize="sm">
              Critical
            </SliderMark>
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <Tooltip
              hasArrow
              bg="brand.500"
              color="white"
              placement="top"
              isOpen={portShowTooltip}
              label={`${Math.round(surveyData.port_weight * 100)}%`}
            >
              <SliderThumb boxSize={6} />
            </Tooltip>
          </Slider>
        </FormControl>
        
        {/* Ecosystem Weight */}
        <FormControl>
          <FormLabel fontSize="lg" fontWeight="bold">
            <HStack>
              <Icon as={FaPuzzlePiece} color="purple.500" />
              <Text>Maturity of tool-chain & ecosystem</Text>
            </HStack>
          </FormLabel>
          <FormHelperText mb={2}>
            How important is having a mature ecosystem with good tools and documentation?
          </FormHelperText>
          
          <Slider
            id="eco-slider"
            defaultValue={surveyData.eco_weight * 5}
            min={1}
            max={5}
            step={1}
            onChange={handleEcoWeightChange}
            onMouseEnter={() => setEcoShowTooltip(true)}
            onMouseLeave={() => setEcoShowTooltip(false)}
            mt={6}
            mb={10}
          >
            <SliderMark value={1} mt={2} ml={-2.5} fontSize="sm">
              Low
            </SliderMark>
            <SliderMark value={5} mt={2} ml={-10} fontSize="sm">
              Critical
            </SliderMark>
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <Tooltip
              hasArrow
              bg="brand.500"
              color="white"
              placement="top"
              isOpen={ecoShowTooltip}
              label={`${Math.round(surveyData.eco_weight * 100)}%`}
            >
              <SliderThumb boxSize={6} />
            </Tooltip>
          </Slider>
        </FormControl>

        {/* Vendor lock-in tolerance */}
        <FormControl>
          <FormLabel fontSize="lg" fontWeight="bold">
            Tolerance for vendor lock-in
          </FormLabel>
          <FormHelperText mb={2}>
            How accepting are you of being tied to a specific vendor's ecosystem?
          </FormHelperText>
          
          <Slider
            id="lockin-slider"
            defaultValue={surveyData.lockin_tolerance * 100}
            min={0}
            max={100}
            step={10}
            onChange={handleLockinToleranceChange}
            onMouseEnter={() => setLockinShowTooltip(true)}
            onMouseLeave={() => setLockinShowTooltip(false)}
            mt={6}
            mb={10}
          >
            <SliderMark value={0} mt={2} ml={-2.5} fontSize="sm">
              Need freedom
            </SliderMark>
            <SliderMark value={100} mt={2} ml={-24} fontSize="sm">
              Single vendor fine
            </SliderMark>
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <Tooltip
              hasArrow
              bg="brand.500"
              color="white"
              placement="top"
              isOpen={lockinShowTooltip}
              label={`${Math.round(surveyData.lockin_tolerance * 100)}%`}
            >
              <SliderThumb boxSize={6} />
            </Tooltip>
          </Slider>
        </FormControl>
        
        {/* GPU programming skill level */}
        <FormControl mt={4}>
          <FormLabel fontSize="lg" fontWeight="bold">
            <HStack>
              <Icon as={FaCode} color="blue.500" />
              <Text>Team's expertise in GPU kernel coding</Text>
            </HStack>
          </FormLabel>
          <FormHelperText mb={4}>
            How would you rate your team's experience with GPU programming?
          </FormHelperText>
          
          <ButtonGroup 
            isAttached={false} 
            variant="outline" 
            size={{ base: "sm", md: "md" }} 
            mt={2}
            display="flex"
            flexWrap="wrap"
            gap={2}
          >
            <Button
              onClick={() => handleSkillLevelChange(0)}
              isActive={surveyData.gpu_skill_level === 0}
              colorScheme={surveyData.gpu_skill_level === 0 ? "brand" : "gray"}
              borderRadius="md"
              px={{ base: 2, sm: 3, md: 4 }}
              flexGrow={{ base: 1, sm: 0 }}
              minW={{ base: "auto", sm: "80px" }}
            >
              None
            </Button>
            <Button
              onClick={() => handleSkillLevelChange(1)}
              isActive={surveyData.gpu_skill_level === 1}
              colorScheme={surveyData.gpu_skill_level === 1 ? "brand" : "gray"}
              borderRadius="md"
              px={{ base: 2, sm: 3, md: 4 }}
              flexGrow={{ base: 1, sm: 0 }}
              minW={{ base: "auto", sm: "80px" }}
            >
              Basic
            </Button>
            <Button
              onClick={() => handleSkillLevelChange(2)}
              isActive={surveyData.gpu_skill_level === 2}
              colorScheme={surveyData.gpu_skill_level === 2 ? "brand" : "gray"}
              borderRadius="md"
              px={{ base: 2, sm: 3, md: 4 }}
              flexGrow={{ base: 1, sm: 0 }}
              minW={{ base: "auto", sm: "80px" }}
            >
              Intermediate
            </Button>
            <Button
              onClick={() => handleSkillLevelChange(3)}
              isActive={surveyData.gpu_skill_level === 3}
              colorScheme={surveyData.gpu_skill_level === 3 ? "brand" : "gray"}
              borderRadius="md"
              px={{ base: 2, sm: 3, md: 4 }}
              flexGrow={{ base: 1, sm: 0 }}
              minW={{ base: "auto", sm: "80px" }}
            >
              Expert
            </Button>
          </ButtonGroup>
        </FormControl>
        
        {/* Programming model preferences */}
        <FormControl mt={6}>
          <FormLabel fontSize="lg" fontWeight="bold">
            Preferred programming model(s)
          </FormLabel>
          <FormHelperText mb={4}>
            Select the programming models your team is most comfortable with.
          </FormHelperText>
          
          <Wrap spacing={{ base: 2, sm: 4 }} direction={{ base: "column", sm: "row" }}>
            <WrapItem width={{ base: "100%", sm: "auto" }}>
              <Tag 
                size={{ base: "md", md: "lg" }}
                variant={surveyData.pref_directives === 1 ? 'solid' : 'outline'} 
                colorScheme={surveyData.pref_directives === 1 ? 'brand' : 'gray'}
                cursor="pointer"
                onClick={toggleDirectives}
                px={3}
                py={2}
                width={{ base: "100%", sm: "auto" }}
                justifyContent={{ base: "center", sm: "flex-start" }}
              >
                <TagLeftIcon as={GrDirections} />
                <TagLabel>Pragmas (OpenMP/OpenACC)</TagLabel>
              </Tag>
            </WrapItem>
            <WrapItem width={{ base: "100%", sm: "auto" }}>
              <Tag 
                size={{ base: "md", md: "lg" }}
                variant={surveyData.pref_kernels === 1 ? 'solid' : 'outline'} 
                colorScheme={surveyData.pref_kernels === 1 ? 'brand' : 'gray'}
                cursor="pointer"
                onClick={toggleKernels}
                px={3}
                py={2}
                width={{ base: "100%", sm: "auto" }}
                justifyContent={{ base: "center", sm: "flex-start" }}
              >
                <TagLeftIcon as={VscSymbolNamespace} />
                <TagLabel>Kernel API (CUDA/HIP/OpenCL)</TagLabel>
              </Tag>
            </WrapItem>
          </Wrap>
        </FormControl>
        
        {/* Framework logos section */}
        <Box mt={6}>
          <Text fontSize="md" fontWeight="medium" mb={3}>Supported Frameworks:</Text>
          <Wrap spacing={6}>
            <WrapItem>
              <Box maxW="80px">
                <img src="/images/cuda-logo.png" alt="CUDA" width="100%" style={{ borderRadius: '4px' }} />
              </Box>
            </WrapItem>
            <WrapItem>
              <Box maxW="80px">
                <img src="/images/opencl-logo.png" alt="OpenCL" width="100%" style={{ borderRadius: '4px' }} />
              </Box>
            </WrapItem>
            <WrapItem>
              <Box maxW="80px">
                <img src="/images/sycl-logo.png" alt="SYCL" width="100%" style={{ borderRadius: '4px' }} />
              </Box>
            </WrapItem>
            <WrapItem>
              <Box maxW="80px">
                <img src="/images/openmp-logo.png" alt="OpenMP" width="100%" style={{ borderRadius: '4px' }} />
              </Box>
            </WrapItem>
            <WrapItem>
              <Box maxW="80px">
                <img src="/images/openacc-logo.png" alt="OpenACC" width="100%" style={{ borderRadius: '4px' }} />
              </Box>
            </WrapItem>
            <WrapItem>
              <Box maxW="80px">
                <img src="/images/kokkos-logo.png" alt="Kokkos" width="100%" style={{ borderRadius: '4px' }} />
              </Box>
            </WrapItem>
            <WrapItem>
              <Box maxW="80px">
                <img src="/images/raja-logo.webp" alt="RAJA" width="100%" style={{ borderRadius: '4px' }} />
              </Box>
            </WrapItem>
            <WrapItem>
              <Box maxW="80px">
                <img src="/images/alpaka-logo.png" alt="ALPAKA" width="100%" style={{ borderRadius: '4px' }} />
              </Box>
            </WrapItem>
            <WrapItem>
              <Box maxW="80px">
                <img src="/images/hip-bg.png" alt="HIP" width="100%" style={{ borderRadius: '4px' }} />
              </Box>
            </WrapItem>
          </Wrap>
        </Box>
        
        <HStack justifyContent="space-between" width="100%" mt={8}>
          <Button
            size="lg"
            variant="outline"
            leftIcon={<FaArrowLeft />}
            onClick={onPrev}
          >
            Back
          </Button>
          <Button
            size="lg"
            colorScheme="brand"
            rightIcon={<FaArrowRight />}
            onClick={onNext}
          >
            Next
          </Button>
        </HStack>
      </VStack>
    </Box>
  );
} 