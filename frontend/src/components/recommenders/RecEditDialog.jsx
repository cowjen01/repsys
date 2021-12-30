import React, { useMemo, useCallback } from 'react';
import { Formik, Field } from 'formik';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  Typography,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import {
  addRecommender,
  recommendersSelector,
  updateRecommender,
} from '../../reducers/recommenders';
import { recEditDialogSelector, closeRecEditDialog, openSnackbar } from '../../reducers/dialogs';
import { TextField, SelectField, CheckboxField } from '../fields';
import { configStatusSelector, modelsSelector } from '../../reducers/config';

function RecEditDialog() {
  const dialog = useSelector(recEditDialogSelector);
  const dispatch = useDispatch();
  const configStatus = useSelector(configStatusSelector);
  const models = useSelector(modelsSelector);
  const recommenders = useSelector(recommendersSelector);

  const handleClose = () => {
    dispatch(closeRecEditDialog());
  };

  const handleSubmit = useCallback(
    (values) => {
      const data = {
        ...values,
        itemsPerPage: parseInt(values.itemsPerPage, 10),
      };
      if (!values.id) {
        dispatch(addRecommender(data));
      } else {
        dispatch(updateRecommender(data));
      }
      dispatch(
        openSnackbar({
          message: 'All settings applied!',
        })
      );
      handleClose();
    },
    [dispatch]
  );

  const initialValues = useMemo(() => {
    if (configStatus !== 'succeeded') {
      return null;
    }

    const defaultParams = Object.fromEntries(
      models.map((m) => [m.name, Object.fromEntries(m.params.map((a) => [a.name, a.default]))])
    );

    if (dialog.index === null) {
      return {
        title: 'New bar',
        itemsPerPage: 4,
        itemsLimit: 20,
        model: models[0].name,
        modelParams: defaultParams,
      };
    }

    const data = recommenders[dialog.index];

    if (!defaultParams[data.model]) {
      return {
        ...data,
        model: '',
      };
    }

    return data;
  }, [dialog, configStatus]);

  return (
    <Dialog open={dialog.open} fullWidth maxWidth="sm" onClose={handleClose}>
      <DialogTitle>Recommender settings</DialogTitle>
      {configStatus === 'succeeded' ? (
        <Formik
          initialValues={initialValues}
          validate={(values) => {
            const errors = {};
            const requiredMessage = 'This field is required.';
            if (!values.title) {
              errors.title = requiredMessage;
            }
            if (!values.itemsLimit) {
              errors.itemsLimit = requiredMessage;
            }
            if (!values.model) {
              errors.model = requiredMessage;
            }
            return errors;
          }}
          onSubmit={(values, { setSubmitting }) => {
            handleSubmit(values);
            setSubmitting(false);
          }}
        >
          {({ submitForm, isSubmitting, values }) => {
            const model = useMemo(
              () => models.find((m) => m.name === values.model),
              [values.model]
            );
            return (
              <>
                <DialogContent>
                  <Grid container direction="column" spacing={2}>
                    <Grid item>
                      <Typography variant="subtitle2" component="div">
                        Appearance
                      </Typography>
                      <Field name="title" label="Title" fullWidth component={TextField} />
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Field
                            name="itemsPerPage"
                            label="Items per page"
                            component={SelectField}
                            options={[1, 3, 4].map((i) => ({ label: i, value: i }))}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <Field
                            name="itemsLimit"
                            label="Max number of items"
                            component={TextField}
                            type="number"
                          />
                        </Grid>
                      </Grid>
                    </Grid>
                    <Grid item>
                      <Typography variant="subtitle2" component="div">
                        Model configuration
                      </Typography>
                      <Field
                        name="model"
                        label="Model"
                        component={SelectField}
                        options={[...models.map((m) => ({ label: m.name, value: m.name }))]}
                        displayEmpty
                      />
                      {model &&
                        model.params &&
                        model.params.map((a) => {
                          const name = `modelParams.${values.model}.${a.name}`;
                          const props = {
                            name,
                            label: a.label,
                          };
                          if (a.type === 'select') {
                            return (
                              <Field
                                key={a.name}
                                component={SelectField}
                                options={a.options.map((b) => ({ label: b, value: b }))}
                                {...props}
                              />
                            );
                          }
                          if (a.type === 'bool') {
                            return <Field key={a.name} component={CheckboxField} {...props} />;
                          }
                          return (
                            <Field key={a.name} component={TextField} type={a.type} {...props} />
                          );
                        })}
                    </Grid>
                  </Grid>
                </DialogContent>
                <DialogActions>
                  <Button onClick={handleClose}>Close</Button>
                  <Button onClick={submitForm} disabled={isSubmitting} autoFocus>
                    Save
                  </Button>
                </DialogActions>
              </>
            );
          }}
        </Formik>
      ) : (
        <DialogContent>Loading ...</DialogContent>
      )}
    </Dialog>
  );
}

export default RecEditDialog;
