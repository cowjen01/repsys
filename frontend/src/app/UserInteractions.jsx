import React from 'react';
import pt from 'prop-types';
import List from '@mui/material/List';
import { Typography, Box, Paper, Button } from '@mui/material';
import Skeleton from '@mui/material/Skeleton';
import Chip from '@mui/material/Chip';
import Grid from '@mui/material/Grid';
import TouchAppIcon from '@mui/icons-material/TouchApp';
import ListSubheader from '@mui/material/ListSubheader';
import FilterListIcon from '@mui/icons-material/FilterList';

import { getRequest } from './api';
import ItemListView from './ItemListView';

function UserInteractions({ selectedUser, onInteractionsDelete, customInteractions }) {
  const { items: userHistoryData, isLoading: isUserHistoryLoading } = getRequest('/interactions', {
    user: selectedUser ? selectedUser.id : null,
  });

  const interactions = customInteractions.length > 0 ? customInteractions : userHistoryData;

  return (
    <Box sx={{ marginTop: 2 }}>
      {customInteractions.length > 0 && (
        <Chip
          // color="primary"
          // size="small"
          sx={{ marginBottom: 2 }}
          onDelete={onInteractionsDelete}
          icon={<FilterListIcon />}
          label="Custom interactions"
        />
      )}

      {!isUserHistoryLoading ? (
        <Paper>
          <List
            subheader={<ListSubheader>Interactions history ({interactions.length})</ListSubheader>}
            sx={{
              width: '100%',
              position: 'relative',
              overflow: 'auto',
              maxHeight: 380,
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
        <Skeleton variant="rectangular" height={380} width="100%" />
      )}
    </Box>
  );
}

export default UserInteractions;
