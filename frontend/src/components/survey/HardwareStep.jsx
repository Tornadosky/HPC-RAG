import {
  Box,
  Heading,
  FormControl,
  FormLabel,
  FormHelperText,
  Checkbox,
  CheckboxGroup,
  Stack,
  RadioGroup,
  Radio,
  Button,
  VStack,
  useColorModeValue,
  Divider,
  SimpleGrid,
  Icon,
  Flex,
  Text
} from '@chakra-ui/react';
import { FaDesktop, FaMicrochip, FaArrowRight } from 'react-icons/fa';
import { SiNvidia, SiAmd } from 'react-icons/si';
import { useSurvey } from '../../context/SurveyContext';

export default function HardwareStep({ onNext }) {
  const { surveyData, updateSurveyData } = useSurvey();
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  // Handle hardware checkbox changes
  const handleHardwareChange = (values) => {
    const hwData = {
      hw_cpu: values.includes('cpu') ? 1 : 0,
      hw_nvidia: values.includes('nvidia') ? 1 : 0,
      hw_amd: values.includes('amd') ? 1 : 0,
      hw_other: values.includes('other') ? 1 : 0
    };
    updateSurveyData(hwData);
  };

  // Get current selected hardware as array for checkbox group
  const getSelectedHardware = () => {
    const selected = [];
    if (surveyData.hw_cpu === 1) selected.push('cpu');
    if (surveyData.hw_nvidia === 1) selected.push('nvidia');
    if (surveyData.hw_amd === 1) selected.push('amd');
    if (surveyData.hw_other === 1) selected.push('other');
    return selected;
  };

  // Handle cross-vendor radio change
  const handleCrossVendorChange = (value) => {
    updateSurveyData({ need_cross_vendor: parseInt(value) });
  };

  // Check if at least one hardware option is selected
  const isHardwareSelected = surveyData.hw_cpu === 1 || 
                            surveyData.hw_nvidia === 1 || 
                            surveyData.hw_amd === 1 || 
                            surveyData.hw_other === 1;

  // Show cross-vendor question only if multiple hardware types are selected
  const multipleHardwareSelected = 
    [surveyData.hw_cpu, surveyData.hw_nvidia, surveyData.hw_amd, surveyData.hw_other]
      .filter(val => val === 1).length > 1;

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
          Step 1: Hardware Configuration
        </Heading>
        
        <Divider />
        
        <FormControl isRequired>
          <FormLabel fontSize="lg" fontWeight="bold">
            Which compute devices must your code run on?
          </FormLabel>
          <FormHelperText mb={4}>
            Select all hardware types that your application needs to support.
          </FormHelperText>
          
          <CheckboxGroup 
            colorScheme="brand" 
            defaultValue={getSelectedHardware()} 
            onChange={handleHardwareChange}
          >
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4} width="100%">
              <HardwareOption 
                value="cpu" 
                icon={FaDesktop} 
                label="CPU" 
                description="Traditional multi-core processors"
              />
              <HardwareOption 
                value="nvidia" 
                icon={SiNvidia} 
                label="NVIDIA GPU" 
                description="CUDA-compatible GPUs"
              />
              <HardwareOption 
                value="amd" 
                icon={SiAmd} 
                label="AMD GPU" 
                description="ROCm-compatible GPUs"
              />
              <HardwareOption 
                value="other" 
                icon={FaMicrochip} 
                label="FPGA / Other" 
                description="FPGAs or other accelerators"
              />
            </SimpleGrid>
          </CheckboxGroup>
        </FormControl>

        {multipleHardwareSelected && (
          <FormControl mt={6}>
            <FormLabel fontSize="lg" fontWeight="bold">
              If you chose more than one vendor, must a single binary run on all of them?
            </FormLabel>
            <FormHelperText mb={4}>
              This affects whether we prioritize cross-platform solutions.
            </FormHelperText>
            
            <RadioGroup 
              onChange={handleCrossVendorChange} 
              value={surveyData.need_cross_vendor.toString()}
              colorScheme="brand"
            >
              <Stack direction="column" spacing={4}>
                <Radio value="1">Yes, need cross-vendor support</Radio>
                <Radio value="0">No, can compile separately for each target</Radio>
              </Stack>
            </RadioGroup>
          </FormControl>
        )}
        
        <Button
          mt={8}
          size="lg"
          colorScheme="brand"
          isDisabled={!isHardwareSelected}
          rightIcon={<FaArrowRight />}
          onClick={onNext}
          alignSelf="flex-end"
        >
          Next
        </Button>
      </VStack>
    </Box>
  );
}

function HardwareOption({ value, icon, label, description }) {
  return (
    <Checkbox value={value} mb={2}>
      <Flex align="center">
        <Icon as={icon} boxSize={5} mr={2} />
        <Box>
          <Text fontWeight="bold">{label}</Text>
          <Text fontSize="sm" color="gray.500">{description}</Text>
        </Box>
      </Flex>
    </Checkbox>
  );
} 