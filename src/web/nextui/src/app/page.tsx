import './Home.css';

import { redirect } from 'next/navigation';

import { IS_RUNNING_LOCALLY } from '@/constants';

export default function Page() {
  if (IS_RUNNING_LOCALLY) {
    redirect('/eval');
  } else {
    redirect('/setup');
  }
}
