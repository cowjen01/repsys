import React, { useMemo } from 'react';
import { TextField, Autocomplete, Box, Chip } from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import FilterListIcon from '@mui/icons-material/FilterList';

import {
  sessionRecordingSelector,
  setSelectedUser,
  customInteractionsSelector,
  selectedUserSelector,
  setCustomInteractions,
} from '../../reducers/root';
import { usersSelector, usersStatusSelector } from '../../reducers/users';
import UserOptionsList from './UserOptionsList';
import UserInteractionsList from './UserInteractionsList';

function UserPanel() {
  const dispatch = useDispatch();
  const sessionRecord = useSelector(sessionRecordingSelector);
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const usersData = useSelector(usersSelector);
  const usersStatus = useSelector(usersStatusSelector);

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
        loading={usersStatus === 'loading'}
        onChange={handleUserSelect}
        options={usersData}
        getOptionLabel={(user) => `User ${user}`}
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
