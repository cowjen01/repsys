import React from 'react';
import pt from 'prop-types';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import { closeTutorialDialog, tutorialDialogSelector } from '../reducers/dialogs';
import { addSeenTutorial } from '../reducers/app';

function TutorialContent({ tutorial }) {
  if (tutorial === 'previews') {
    return (
      <iframe
        width="560"
        height="315"
        src="https://www.youtube.com/embed/dsUXAEzaC3Q"
        title="YouTube video player"
        frameBorder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
      />
    );
  }
  return null;
}

TutorialContent.propTypes = {
  tutorial: pt.string,
};

TutorialContent.defaultProps = {
  tutorial: '',
};

function TutorialDialog() {
  const dialog = useSelector(tutorialDialogSelector);

  const dispatch = useDispatch();

  const handleClose = () => {
    dispatch(closeTutorialDialog());
    dispatch(addSeenTutorial(dialog.tutorial));
  };

  const getTutorialTitle = () => {
    if (dialog.tutorial === 'previews') {
      return 'Recommendation Previews';
    }
    return '';
  };

  return (
    <Dialog open={dialog.open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Tutorial - {getTutorialTitle()}</DialogTitle>
      <DialogContent>
        <DialogContentText>
          <TutorialContent tutorial={dialog.tutorial} />
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default TutorialDialog;
