import { useState, useRef, useEffect } from 'react';
import {
  Box,
  Input,
  Button,
  VStack,
  HStack,
  Text,
  Heading,
  useColorModeValue,
  Spinner,
  Collapse,
  Flex,
  Badge,
  IconButton,
  Divider,
  useToast
} from '@chakra-ui/react';
import { keyframes } from '@emotion/react';
import { ChevronRightIcon, ChatIcon, CloseIcon } from '@chakra-ui/icons';
import { motion } from 'framer-motion';

const MotionBox = motion(Box);

// Define a blink keyframe animation
const blinkAnimation = keyframes`
  0% { opacity: 0.4; }
  50% { opacity: 1; }
  100% { opacity: 0.4; }
`;

export default function ChatWidget({ userProfile, frameworkRanking }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentResponse, setCurrentResponse] = useState('');
  const messagesEndRef = useRef(null);
  const toast = useToast();
  
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const userBgColor = useColorModeValue('brand.50', 'brand.900');
  const botBgColor = useColorModeValue('gray.50', 'gray.700');
  
  // Auto-scroll to bottom of chat
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, currentResponse]);
  
  // Parse streaming response data
  const parseSSEData = (data) => {
    if (data.startsWith('data: ')) {
      return data.substring(6);
    }
    return data;
  };
  
  // Enhanced formatText function to handle more text formatting cases
  const formatText = (text) => {
    if (!text) return '';
    
    // Known acronyms that should not have spaces
    const acronyms = {
      'H PC': 'HPC',
      'G PU': 'GPU',
      'C PU': 'CPU',
      'AP I': 'API',
      'A I': 'AI',
      'ML ': 'ML',
      'CU DA': 'CUDA'
    };
    
    // Known framework names to fix
    const frameworkNames = {
      'AL PA KA': 'ALPAKA',
      'ALPA KA': 'ALPAKA',
      'AL PAKA': 'ALPAKA',
      'Open ACC': 'OpenACC',
      'Open MP': 'OpenMP',
      'Open CL': 'OpenCL',
      'Std Par': 'StdPar',
      'Kokkos': 'Kokkos',
      'RAJA': 'RAJA',
      'TBB': 'TBB',
      'SYCL': 'SYCL',
      'Hip PCC': 'HipPCC',
      'Hip PCL': 'HipPCL'
    };
    
    let formattedText = text;
    
    // Fix framework names (first, as they're more specific)
    Object.entries(frameworkNames).forEach(([incorrect, correct]) => {
      const regex = new RegExp(`\\b${incorrect.replace(/\s+/g, '\\s+')}\\b`, 'gi');
      formattedText = formattedText.replace(regex, correct);
    });
    
    // Fix acronyms
    Object.entries(acronyms).forEach(([incorrect, correct]) => {
      const regex = new RegExp(`\\b${incorrect.replace(/\s+/g, '\\s+')}\\b`, 'gi');
      formattedText = formattedText.replace(regex, correct);
    });
    
    // Fix possessive apostrophes with spaces (user 's → user's)
    formattedText = formattedText.replace(/(\w)\s+'s\b/g, "$1's");
    
    // Fix contracted words with spaces (don 't → don't)
    formattedText = formattedText.replace(/(\w)\s+'(t|s|ll|re|ve|d|m)\b/g, "$1'$2");
    
    // Remove spaces before punctuation
    formattedText = formattedText.replace(/\s+([.,;:!?)\]}])/g, '$1');
    
    // Ensure single space after punctuation
    formattedText = formattedText.replace(/([.,;:!?)\]}])(?!\s|$)/g, '$1 ');
    
    // Fix brackets with spaces inside ( text ) → (text)
    formattedText = formattedText
      .replace(/\(\s+/g, '(')
      .replace(/\s+\)/g, ')')
      .replace(/\[\s+/g, '[')
      .replace(/\s+\]/g, ']')
      .replace(/\{\s+/g, '{')
      .replace(/\s+\}/g, '}');
    
    // Fix decimal numbers (remove spaces between digits and decimal point)
    formattedText = formattedText.replace(/(\d)\s+\.\s+(\d)/g, '$1.$2');
    
    // Fix other decimal formats (0. 5 → 0.5)
    formattedText = formattedText.replace(/(\d)\.\s+(\d)/g, '$1.$2');
    
    // Fix reference formats like [ 1 ] → [1]
    formattedText = formattedText.replace(/\[\s+(\d+)\s+\]/g, '[$1]');
    
    // Fix tuple/parentheses formats like ( 2 ) → (2)
    formattedText = formattedText.replace(/\(\s+(\d+)\s+\)/g, '($1)');
    
    // Fix number formatting with commas (1 , 000 → 1,000)
    formattedText = formattedText.replace(/(\d)\s+,\s+(\d)/g, '$1,$2');
    
    // Fix hyphens used as word joiners (log -based → log-based)
    formattedText = formattedText.replace(/(\w)\s+-\s*(\w)/g, '$1-$2');
    
    // Fix formatting for ordinals (2 nd → 2nd)
    formattedText = formattedText.replace(/(\d+)\s+(st|nd|rd|th)\b/g, '$1$2');
    
    // Fix percentage symbol spacing (6.7 % → 6.7%)
    formattedText = formattedText.replace(/(\d+(?:\.\d+)?)\s+%/g, '$1%');
    
    // Convert decimal scores to percentages (0.067 → 6.7%)
    formattedText = formattedText.replace(/\b0\.(\d{2})(\d?)\b/g, (match, p1, p2) => {
      const percentage = parseFloat(`0.${p1}${p2 || ''}`) * 100;
      return `${Math.round(percentage * 10) / 10}%`;
    });
    
    // Double spaces to single spaces
    formattedText = formattedText.replace(/\s+/g, ' ');
    
    return formattedText;
  };
  
  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;
    
    // Add user message to chat
    const userMessage = { role: 'user', content: inputValue };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setCurrentResponse('');
    
    try {
      // Prepare the payload for the API
      const payload = {
        question: inputValue,
        profile: userProfile,
        ranking: frameworkRanking
      };
      
      // Add a placeholder for the streaming response
      setMessages(prevMessages => [...prevMessages, { 
        role: 'assistant', 
        content: '', 
        isStreaming: true 
      }]);
      
      // Stream the response using fetch with ReadableStream
      const streamResponse = await fetch('/api/chat?stream=true', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      if (!streamResponse.ok) {
        throw new Error(`HTTP error! status: ${streamResponse.status}`);
      }
      
      // Get a reader from the response body stream
      const reader = streamResponse.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let responseText = '';
      
      // Process the stream
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          // Stream is complete, fetch citations and finalize
          await fetchCitations(responseText, payload);
          break;
        }
        
        // Decode the received chunk
        const chunk = decoder.decode(value, { stream: true });
        // Parse any SSE format (data: token\n\n)
        const lines = chunk.split('\n').filter(line => line.trim() !== '');
        
        for (const line of lines) {
          if (line.startsWith('data:')) {
            const token = parseSSEData(line);
            
            // Only add space if needed for proper spacing
            if (responseText && 
                !responseText.endsWith(' ') && 
                !token.startsWith(' ') && 
                !token.match(/^[.,;:!?)\]}\-–—]/)) {
              responseText += ' ';
            }
            
            responseText += token;
            
            // Format text for display
            const formattedText = formatText(responseText);
            
            // Update the current message with the streamed content
            setCurrentResponse(formattedText);
            setMessages(prevMessages => {
              const updated = [...prevMessages];
              const assistantIndex = updated.length - 1;
              if (assistantIndex >= 0 && updated[assistantIndex].isStreaming) {
                updated[assistantIndex] = { 
                  role: 'assistant', 
                  content: formattedText,
                  isStreaming: true 
                };
              }
              return updated;
            });
          } else if (line.includes('event: done')) {
            // Handle done event
            break;
          }
        }
      }
    } catch (error) {
      console.error('Error in chat stream:', error);
      toast({
        title: 'Error',
        description: 'Failed to get a response. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      
      // Clean up on error
      setMessages(prevMessages => {
        // Remove the streaming message if it exists
        return prevMessages.filter(msg => !msg.isStreaming);
      });
      setCurrentResponse('');
      setIsLoading(false);
    }
  };
  
  // Fetch citations and finalize the response
  const fetchCitations = async (responseText, payload) => {
    try {
      const citationsResponse = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      if (citationsResponse.ok) {
        const data = await citationsResponse.json();
        
        // Format the text before displaying it
        const formattedText = formatText(responseText || data.answer);
        
        // Update the message with the final text and citations
        setMessages(prevMessages => {
          const updated = [...prevMessages];
          const assistantIndex = updated.findIndex(msg => msg.isStreaming);
          if (assistantIndex >= 0) {
            updated[assistantIndex] = { 
              role: 'assistant', 
              content: formattedText,
              citations: data.citations || [],
              isStreaming: false
            };
          }
          return updated;
        });
      }
      
      setCurrentResponse('');
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching citations:', error);
      setIsLoading(false);
      
      // Make sure to set isStreaming to false on error
      setMessages(prevMessages => {
        return prevMessages.map(msg => 
          msg.isStreaming ? {...msg, isStreaming: false} : msg
        );
      });
    }
  };
  
  // Handle Enter key press
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };
  
  // Simple cursor that blinks
  const TypingIndicator = () => (
    <Box
      as="span"
      display="inline-block"
      ml={1}
      height="16px"
      opacity={0.7}
      sx={{
        animation: `${blinkAnimation} 1.2s infinite`,
      }}
    >
      ▌
    </Box>
  );
  
  return (
    <MotionBox
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      width="100%"
      mt={6}
    >
      {/* Chat toggle button when closed */}
      {!isOpen && (
        <Button
          leftIcon={<ChatIcon />}
          colorScheme="brand"
          variant="outline"
          onClick={() => setIsOpen(true)}
          width="100%"
        >
          Ask questions about these frameworks
        </Button>
      )}
      
      {/* Chat interface when open */}
      <Collapse in={isOpen} animateOpacity>
        <Box 
          borderWidth="1px" 
          borderColor={borderColor} 
          borderRadius="lg"
          bg={bgColor}
          boxShadow="md"
          overflow="hidden"
        >
          {/* Chat header */}
          <Flex 
            justifyContent="space-between" 
            alignItems="center" 
            p={4} 
            borderBottomWidth="1px"
            borderColor={borderColor}
            bg="brand.500"
            color="white"
          >
            <Heading size="md">Framework Assistant</Heading>
            <IconButton
              icon={<CloseIcon />}
              variant="ghost"
              size="sm"
              color="white"
              _hover={{ bg: 'brand.600' }}
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
            />
          </Flex>
          
          {/* Chat messages */}
          <Box 
            p={4} 
            height="350px" 
            overflowY="auto"
            css={{
              '&::-webkit-scrollbar': {
                width: '4px',
              },
              '&::-webkit-scrollbar-track': {
                width: '6px',
              },
              '&::-webkit-scrollbar-thumb': {
                background: useColorModeValue('gray.300', 'gray.600'),
                borderRadius: '24px',
              },
            }}
          >
            {messages.length === 0 ? (
              <Text color="gray.500" textAlign="center" py={4}>
                Ask me about HPC frameworks, programming models, or hardware compatibility.
              </Text>
            ) : (
              <VStack spacing={4} align="stretch">
                {messages.map((message, index) => (
                  <MotionBox
                    key={index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    alignSelf={message.role === 'user' ? 'flex-end' : 'flex-start'}
                    maxWidth="80%"
                  >
                    <Box
                      p={3}
                      borderRadius="lg"
                      bg={message.role === 'user' ? userBgColor : botBgColor}
                      boxShadow="sm"
                    >
                      <Text>
                        {message.content}
                        {message.isStreaming && <TypingIndicator />}
                      </Text>
                      
                      {/* Show citations if available */}
                      {message.citations && message.citations.length > 0 && (
                        <Box mt={2}>
                          <Divider mb={2} />
                          <Text fontSize="xs" fontWeight="bold" mb={1}>Sources:</Text>
                          <Flex wrap="wrap" gap={1}>
                            {message.citations.map((citation, idx) => (
                              <Badge key={idx} colorScheme="blue" fontSize="xs">
                                {citation}
                              </Badge>
                            ))}
                          </Flex>
                        </Box>
                      )}
                    </Box>
                  </MotionBox>
                ))}
                <div ref={messagesEndRef} />
              </VStack>
            )}
            
            {isLoading && messages.length === 0 && (
              <Box textAlign="center" mt={4}>
                <Spinner size="sm" colorScheme="brand" mr={2} />
                <Text fontSize="sm" color="gray.500" display="inline">
                  Thinking...
                </Text>
              </Box>
            )}
          </Box>
          
          {/* Chat input */}
          <HStack 
            p={4} 
            borderTopWidth="1px"
            borderColor={borderColor}
            spacing={2}
          >
            <Input
              placeholder="Ask about frameworks or programming models..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
            />
            <Button
              colorScheme="brand"
              onClick={handleSendMessage}
              isLoading={isLoading}
              disabled={!inputValue.trim() || isLoading}
              rightIcon={<ChevronRightIcon />}
            >
              Send
            </Button>
          </HStack>
        </Box>
      </Collapse>
    </MotionBox>
  );
} 