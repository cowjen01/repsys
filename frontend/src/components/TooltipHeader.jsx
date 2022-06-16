import React from 'react';
import pt from 'prop-types';
import { Typography, Tooltip, IconButton, Stack } from '@mui/material';
import HelpIcon from '@mui/icons-material/Help';

function TooltipHeader({ title, tooltip }) {
  return (
    <Stack direction="row" sx={{ mb: 1 }}>
      <Typography component="div" variant="h6">
        {title}
      </Typography>
      <Tooltip title={tooltip}>
        <IconButton color="primary" size="small">
          <HelpIcon />
        </IconButton>
      </Tooltip>
    </Stack>
  );
}

TooltipHeader.propTypes = {
  title: pt.string.isRequired,
  tooltip: pt.string.isRequired,
};

export default TooltipHeader;
