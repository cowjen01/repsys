import React from 'react';
import { TextField, Autocomplete, Box, Chip } from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import FilterListIcon from '@mui/icons-material/FilterList';

import { getRequest } from '../../api';
import {
  sessionRecordingSelector,
  setSelectedUser,
  customInteractionsSelector,
  selectedUserSelector,
  setCustomInteractions,
} from '../../reducers/root';
import UserOptionsList from './UserOptionsList';
import UserInteractionsList from './UserInteractionsList';

function UserPanel() {
  const dispatch = useDispatch();
  const sessionRecord = useSelector(sessionRecordingSelector);
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);

  const { items: userData, isLoading: isUserLoading } = getRequest('/users');

  const handleUserSelect = (event, user) => {
    dispatch(setSelectedUser(user));
    dispatch(setCustomInteractions([]));
  };

  const handleDelete = () => {
    dispatch(setCustomInteractions([]));
  };

  return (
    <Box sx={{ position: 'sticky', top: '4rem' }}>
      <Autocomplete
        disablePortal
        disabled={sessionRecord}
        value={selectedUser}
        loading={isUserLoading}
        onChange={handleUserSelect}
        isOptionEqualToValue={(option, value) => option.id === value.id}
        options={userData}
        getOptionLabel={(user) => `User ${user.label}`}
        sx={{ width: '100%', marginBottom: 2 }}
        renderInput={(params) => <TextField {...params} variant="filled" label="Selected user" />}
      />
      <UserOptionsList />
      {(selectedUser || customInteractions.length > 0) && (
        <Box sx={{ marginTop: 2 }}>
          {customInteractions.length > 0 && (
            <Chip
              sx={{ marginBottom: 2 }}
              onDelete={handleDelete}
              icon={<FilterListIcon />}
              label="Custom interactions"
            />
          )}
          <UserInteractionsList />
        </Box>
      )}
    </Box>
  );
}

export default UserPanel;
