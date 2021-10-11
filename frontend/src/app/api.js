import { useState, useEffect } from 'react';

export function fetchItems(path) {
  const [items, setItems] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isActive = true;

    fetch(`/api${path}`)
      .then((response) => response.json())
      .then((data) => {
        if (isActive) {
          setItems(data);
        }
        setIsLoading(false);
      })
      .catch((error) => {
        setIsLoading(false);
      });

    return () => {
      isActive = false;
    };
  }, []);

  return { items, isLoading };
}
