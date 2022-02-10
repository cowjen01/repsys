import React from 'react';
import pt from 'prop-types';
import { Box } from '@mui/material';

function TabPanel({ children, value, index, ...other }) {
  return (
    <Box role="tabpanel" hidden={value !== index} {...other}>
      {value === index ? children : null}
    </Box>
  );
}

TabPanel.propTypes = {
  children: pt.element.isRequired,
  value: pt.number.isRequired,
  index: pt.number.isRequired,
};

export default TabPanel;
