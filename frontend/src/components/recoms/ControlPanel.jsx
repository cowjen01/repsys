import React from 'react';
import {
  Box,
  Chip,
  Switch,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Paper,
  Skeleton,
  Typography,
  Alert,
} from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import FilterListIcon from '@mui/icons-material/FilterList';
import RadioButtonCheckedIcon from '@mui/icons-material/RadioButtonChecked';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import BuildIcon from '@mui/icons-material/Build';
import { FixedSizeList } from 'react-window';

import {
  interactiveModeSelector,
  customInteractionsSelector,
  selectedUserSelector,
  setCustomInteractions,
  toggleInteractiveMode,
  buildModeSelector,
  toggleBuildMode,
  setSelectedUser,
} from '../../reducers/app';
import { ItemListView } from '../items';
import { openSnackbar, openUserSelectDialog } from '../../reducers/dialogs';
import { sliceIdentifier } from '../../utils';
import { useGetUserByIDQuery } from '../../api';

const listHeight = 360;

function renderRow({ index, style, data }) {
  const item = data[index];
  return <ItemListView style={style} key={item.id} item={item} />;
}

function UserPanel() {
  const dispatch = useDispatch();
  const interactiveMode = useSelector(interactiveModeSelector);
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const buildMode = useSelector(buildModeSelector);

  const user = useGetUserByIDQuery(selectedUser, {
    skip: !selectedUser,
  });

  const handleDelete = () => {
    dispatch(setCustomInteractions([]));
    dispatch(setSelectedUser(null));
  };

  const handleInteractiveModeChange = () => {
    if (selectedUser) {
      dispatch(toggleInteractiveMode(user.data.interactions));
    } else {
      dispatch(toggleInteractiveMode());
    }
    if (!interactiveMode) {
      dispatch(
        openSnackbar({
          message: 'Interactive mode - click on the items to interact.',
          severity: 'info',
        })
      );
    }
  };

  const handleSelectBtnClick = () => {
    dispatch(openUserSelectDialog());
  };

  const interactions = selectedUser && user.data ? user.data.interactions : customInteractions;

  return (
    <Box sx={{ position: 'sticky', top: '4rem' }}>
      <Paper>
        <List>
          <ListItem disabled={interactiveMode}>
            <ListItemIcon>
              <BuildIcon />
            </ListItemIcon>
            <ListItemText primary="Build Mode" />
            <Switch
              color="secondary"
              edge="end"
              onChange={() => dispatch(toggleBuildMode())}
              checked={buildMode}
            />
          </ListItem>
          <ListItem disabled={buildMode}>
            <ListItemIcon>
              <RadioButtonCheckedIcon />
            </ListItemIcon>
            <ListItemText primary="Interactive Mode" />
            <Switch
              color="secondary"
              edge="end"
              onChange={handleInteractiveModeChange}
              checked={interactiveMode}
            />
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton disabled={interactiveMode || buildMode} onClick={handleSelectBtnClick}>
              <ListItemIcon>
                <PersonSearchIcon />
              </ListItemIcon>
              <ListItemText primary="Interactions Selector" />
            </ListItemButton>
          </ListItem>
        </List>
      </Paper>
      {customInteractions.length > 0 || selectedUser ? (
        <Box sx={{ marginTop: 2 }}>
          {customInteractions.length > 0 && (
            <Chip
              sx={{ marginBottom: 2 }}
              onDelete={handleDelete}
              icon={<FilterListIcon />}
              label="Custom interactions"
            />
          )}
          {selectedUser && (
            <Chip
              sx={{ marginBottom: 2 }}
              onDelete={handleDelete}
              icon={<FilterListIcon />}
              label={`User ${sliceIdentifier(selectedUser)}`}
            />
          )}
          {!user.isFetching ? (
            <Paper>
              <Typography
                sx={{
                  lineHeight: '48px',
                  color: 'text.secondary',
                  pl: '16px',
                }}
                variant="subtitle2"
                component="div"
              >
                Interactions ({interactions.length})
              </Typography>
              <FixedSizeList
                height={listHeight}
                itemData={interactions}
                itemSize={60}
                itemCount={interactions.length}
              >
                {renderRow}
              </FixedSizeList>
            </Paper>
          ) : (
            <Skeleton variant="rectangular" height={listHeight + 48} width="100%" />
          )}
        </Box>
      ) : (
        <Box sx={{ marginTop: 2 }}>
          <Alert severity="warning">No interactions selected.</Alert>
        </Box>
      )}
    </Box>
  );
}

export default UserPanel;
