import React from 'react';
import pt from 'prop-types';
import List from '@mui/material/List';
import { Typography, Box, Paper, Button } from '@mui/material';
import Skeleton from '@mui/material/Skeleton';
import Chip from '@mui/material/Chip';
import TouchAppIcon from '@mui/icons-material/TouchApp';

import { getRequest } from './api';
import ItemListView from './ItemListView';

function UserInteractions({ selectedUser, onInteractionsDelete, customInteractions }) {

  const { items: userHistoryData, isLoading: isUserHistoryLoading } = getRequest('/interactions', {
    user: selectedUser ? selectedUser.id : null,
  });

  const interactions = customInteractions || userHistoryData;

  return (
    <Box sx={{ marginTop: 2 }}>
      <Typography variant="h6" component="div" gutterBottom>
        User Interactions ({interactions.length})
      </Typography>
      {customInteractions && (
        <Chip
          // color="primary"
          size="small"
          sx={{ marginBottom: 2 }}
          onDelete={onInteractionsDelete}
          icon={<TouchAppIcon />}
          label="Custom Interactions"
        />
      )}
      {!isUserHistoryLoading ? (
        <Paper>
          <List
            sx={{
              width: '100%',
              position: 'relative',
              overflow: 'auto',
              maxHeight: 400,
            }}
          >
            {interactions.map((item) => (
              <ItemListView
                image={item.image}
                key={item.id}
                id={item.id}
                title={item.title}
                subtitle={item.subtitle}
              />
            ))}
          </List>
        </Paper>
      ) : (
        <Skeleton variant="rectangular" height={400} width="100%" />
      )}
    </Box>
  );
}

export default UserInteractions;
