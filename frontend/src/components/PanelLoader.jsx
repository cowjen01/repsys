import React from 'react';
import pt from 'prop-types';
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

PanelLoader.propTypes = {
  height: pt.number,
};

PanelLoader.defaultProps = {
  height: 300,
};

export default PanelLoader;
