import React from 'react';
import pt from 'prop-types';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import { Typography, Box, ListItemText, Paper, Button } from '@mui/material';

import { fetchItems } from './api';

function UserPanel({ selectedUser, onUserSelect, onSearchClick }) {
  const { items: userData, isLoading: isUserLoading } = fetchItems('/users');
  const { items: userHistoryData, isLoading: isUserHistoryLoading } = fetchItems('/interactions', {
    user: selectedUser,
  });

  return (
    <Box sx={{ position: 'sticky', top: '4rem' }}>
      <Typography variant="h6" component="div" gutterBottom>
        User Selector
      </Typography>
      <Autocomplete
        disablePortal
        value={selectedUser ? { id: selectedUser } : null}
        onChange={(event, newValue) => {
          if (newValue) {
            onUserSelect(newValue.id);
          } else {
            onUserSelect(null);
          }
        }}
        isOptionEqualToValue={(option, value) => option.id === value.id}
        options={userData}
        getOptionLabel={(user) => `User ${user.id}`}
        sx={{ width: '100%', marginBottom: 2 }}
        renderInput={(params) => <TextField {...params} variant="filled" label="Selected user" />}
      />
      <Button onClick={onSearchClick} startIcon={<PersonSearchIcon />} color="secondary">
        Advanced search
      </Button>
      {selectedUser && (
        <Box sx={{ marginTop: 1 }}>
          <Typography variant="h6" component="div" gutterBottom>
            User Interactions ({userHistoryData.length})
          </Typography>
          <Paper>
            <List
              sx={{
                width: '100%',
                position: 'relative',
                overflow: 'auto',
                maxHeight: 400,
              }}
            >
              {userHistoryData.map((item) => (
                <ListItem key={item.id}>
                  <ListItemText primary={item.title} secondary={item.subtitle} />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Box>
      )}
    </Box>
  );
}

UserPanel.defaultProps = {
  selectedUser: null,
};

UserPanel.propTypes = {
  selectedUser: pt.string,
  onUserSelect: pt.func.isRequired,
  onSearchClick: pt.func.isRequired,
};

export default UserPanel;
