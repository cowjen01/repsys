import React from 'react';
import Box from '@mui/material/Box';
import Skeleton from '@mui/material/Skeleton';

function ItemSkeleton() {
  return (
    <Box sx={{ pt: 0.5, width: '100%' }}>
      <Skeleton variant="rectangular" height={155} />
    </Box>
  );
}

export default ItemSkeleton;
