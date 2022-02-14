import React from 'react';
import pt from 'prop-types';
import { Box, CircularProgress } from '@mui/material';

function PanelLoader({ height }) {
  return (
    <Box
      display="flex"
      textAlign="center"
      height={height}
      alignItems="center"
      justifyContent="center"
    >
      <CircularProgress />
    </Box>
  );
}

PanelLoader.propTypes = {
  height: pt.oneOfType([pt.number, pt.string]),
};

PanelLoader.defaultProps = {
  height: '100%',
};

export default PanelLoader;
