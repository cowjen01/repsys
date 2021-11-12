import React from 'react';
import pt from 'prop-types';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import { Typography, Box, Button } from '@mui/material';

import { getRequest } from './api';
import UserInteractions from './UserInteractions';

function UserPanel({
  selectedUser,
  onUserSelect,
  onSearchClick,
  customInteractions,
  onInteractionsDelete,
}) {
  const { items: userData, isLoading: isUserLoading } = getRequest('/users');

  return (
    <Box sx={{ position: 'sticky', top: '4rem' }}>
      <Typography variant="h6" component="div" gutterBottom>
        User Selector
      </Typography>
      <Autocomplete
        disablePortal
        value={selectedUser}
        onChange={(event, newValue) => {
          onUserSelect(newValue);
        }}
        isOptionEqualToValue={(option, value) => option.id === value.id}
        options={userData}
        getOptionLabel={(user) => `User ${user.label}`}
        sx={{ width: '100%', marginBottom: 2 }}
        renderInput={(params) => <TextField {...params} variant="filled" label="Selected user" />}
      />
      <Button
        onClick={onSearchClick}
        startIcon={<PersonSearchIcon />}
        variant="contained"
        color="secondary"
      >
        More Options
      </Button>
      {(selectedUser || customInteractions) && (
        <UserInteractions
          selectedUser={selectedUser}
          onInteractionsDelete={onInteractionsDelete}
          customInteractions={customInteractions}
        />
      )}
    </Box>
  );
}

UserPanel.defaultProps = {
  selectedUser: null,
};

UserPanel.propTypes = {
  selectedUser: pt.shape({
    id: pt.number,
    label: pt.string,
  }),
  onUserSelect: pt.func.isRequired,
  onSearchClick: pt.func.isRequired,
};

export default UserPanel;
