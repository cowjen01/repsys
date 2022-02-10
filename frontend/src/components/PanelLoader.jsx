import React from 'react';
import { Box, CircularProgress } from '@mui/material';

function PanelLoader({ height }) {
  return (
    <Box
      display="flex"
      textAlign="center"
      minHeight={height}
      alignItems="center"
      justifyContent="center"
    >
      <CircularProgress />
    </Box>
  );
}

PanelLoader.defaultProps = {
  height: 100,
};

export default PanelLoader;
