import { Routes, Route } from 'react-router-dom';
import { Box } from '@chakra-ui/react';
import Header from './components/layout/Header';
import Footer from './components/layout/Footer';
import LandingPage from './pages/LandingPage';
import SurveyPage from './pages/SurveyPage';
import ResultsPage from './pages/ResultsPage';

function App() {
  return (
    <Box minH="100vh" display="flex" flexDirection="column">
      <Header />
      <Box flex="1" as="main">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/survey" element={<SurveyPage />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
      </Box>
      <Footer />
    </Box>
  );
}

export default App; 